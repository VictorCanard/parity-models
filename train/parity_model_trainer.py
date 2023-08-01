import copy
import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import time
import wandb

from base_models.base_model_wrapper import BaseModelWrapper
from datasets.code_dataset import get_dataloaders
from loss.masked_loss import MaskedLoss
import util.stats
import util.util
from util.util import construct, try_cuda


class ParityModelTrainer(object):
    """
    Top-level class which carries out the training of a parity model.
    """

    def __init__(self, config_map, checkpoint_cycle=1):
        """
        Parameters
        ----------
        config_map: dict
            A dictionary containing the full specification of components to be
            used in training. Examples of these may be found in the "conf"
            directory of this project.
        checkpoint_cycle: int
            Number of epochs between model checkpoints
        """
        self.__init_from_config_map(config_map)
        self.erase_mask, self.acc_mask = self.__gen_masks()
        self.checkpoint_cycle = checkpoint_cycle




    def train(self, wandb=False):
        """
        Trains the encoder and decoder as specified via configuration.
        """
        self.base_model.eval()
        while self.cur_epoch < self.final_epoch:
            if not self.only_test:
                # Perform epoch on training set
                self.__epoch(self.train_dataloader, wandb, do_step=True)

                # Perform epoch on validation set
                _, val_recon_acc, _ = self.__epoch(self.val_dataloader, wandb,
                                                   do_step=False)

            # Perform epoch on test dataset
            _, _, _ = self.__epoch(self.test_dataloader, wandb,
                                   do_step=False,
                                   do_print=False)

            if not self.only_test:
                self.__save_current_state(val_recon_acc)
            self.cur_epoch += 1

            # Place functions back on GPU, if necessary
            self.parity_model = try_cuda(self.parity_model)
            self.loss_fn = try_cuda(self.loss_fn)

    def __epoch(self, data_loader, wandb, do_step=False, do_print=True):
        """
        Performs a single epoch of either training or validation.

        Parameters
        ----------
        data_loader:
            The data loader to use for this epoch
        do_step: bool
            Whether to make optimization steps using this data loader.
        do_print: bool
            Whether to print accuracies for this epoch.
        """
        stats = util.stats.StatsTracker()
        label = data_loader.dataset.name

        if label == "train":
            self.parity_model.train()
        else:
            self.parity_model.eval()

        if do_print:
            data_loader = tqdm(data_loader, ascii=True,
                               desc="Epoch {}. {}".format(self.cur_epoch, label))

        for mb_data, mb_labels, mb_true_labels in data_loader:  # Is it ok putting one batch of data on the gpu?
            mb_data = try_cuda(mb_data.view(-1, self.ec_k, mb_data.size(1)))
            mb_labels = try_cuda(
                mb_labels.view(-1, self.ec_k, mb_labels.size(1), mb_labels.size(2)))  # size 2?
            mb_true_labels = try_cuda(mb_true_labels.view(-1, self.ec_k, mb_true_labels.size(1)))

            if do_step:
                if self.train_parity_model: self.parity_model_opt.zero_grad()
                if self.train_encoder: self.encoder_opt.zero_grad()
                if self.train_decoder: self.decoder_opt.zero_grad()

            loss = self.__forward(mb_data, mb_labels, mb_true_labels, stats)

            if do_step:
                loss.backward()
                if self.train_parity_model: self.parity_model_opt.step()
                if self.train_encoder: self.encoder_opt.step()
                if self.train_decoder: self.decoder_opt.step()

            if do_print:
                rloss, rtop1, rtop5 = stats.running_averages()
                data_loader.set_description(
                    "Epoch {}. {}. Top-1={:.4f}, Top-5={:.4f}, Loss={:.4f}".format(
                        self.cur_epoch, label, rtop1, rtop5, rloss))

        epoch_loss, epoch_acc_map = stats.averages()
        outfile_fmt = os.path.join(self.save_dir, label + "_{}.txt")
        epoch_map = epoch_acc_map
        epoch_map["loss"] = epoch_loss
        util.util.write_vals_dict(outfile_fmt, epoch_map, wandb, label)

        top_recon = epoch_acc_map["reconstruction_top1"]
        top_overall = epoch_acc_map["overall_top1"]
        return epoch_loss, top_recon, top_overall 

    def __gen_masks(self):
        """Generates masks to be used when erasing indices, calculating loss,
        and calculating accuracies.

        This method currently assumes that only one element will be erased, and
        thus that ec_r is 1.

        Returns
        -------
            ``torch.autograd.Variable``:
                A mask of all ones but with zeros in the locations of elements
                which should be erased when simulating erasure. There is one
                mask for each possible combination of erased and non-erased
                data units.
                Dimensions: (ec_k, ec_k + ec_r, base_model_output_dim)
            ``torch.autograd.Variable``:
                Same as loss_mask, but with of dimensionality to be used when
                calculating accuracies. There is one mask for each possible
                combination of erased and non-erased data units.
                Dimensions: (ec_k, ec_k)
        """
        base_model_output_dim = self.val_dataloader.dataset.decoder_in_dim()

        # As this method assumes that only one data unit is erased at a given time,
        # the only possible erasure scenarios correspond to when one of the first
        # `ec_r` elements are erased.
        erased_indices = [torch.LongTensor([e]) for e in range(self.ec_k)]

        erase_mask = torch.ones((len(erased_indices),
                                 self.ec_k + self.ec_r,
                                 base_model_output_dim))

        acc_mask = torch.zeros((len(erased_indices), self.ec_k)).byte()

        for i, erased_idx in enumerate(erased_indices):
            i = torch.LongTensor([i])
            erase_mask[i, erased_idx, :] = 0.
            acc_mask[i, erased_idx] = 1

        return try_cuda(erase_mask), try_cuda(acc_mask)

    def __forward(self, mb_data, mb_labels, mb_true_labels, stats):
        """
        Performs a forward pass by encoding `mb_data` together, passing the
        resultant parity through the base model, decoding (simulated)
        unavailable outputs, and calculating loss.
        """
        batch_size = mb_data.size(0)
        seq_len = mb_data.size(2)

        # The encoder encodes across channels. That is, if an input is represented
        # with three channels as RGB inputs, we encode each of the `ec_k` R
        # channels together, each of the `ec_k` G channels together, and each of
        # the `ec_k` B channels together. The resultant parity channels are then
        # concatenated together to form a "parity" image.
        #
        # Here we reshape multi-channel inputs to align them for this encoding.
        if self.train_dataloader.dataset.num_channels > 1:
            # The number of channels must be 3 at this point, based on
            # assertions in the data loading process.
            mb_data = mb_data.view(-1, self.ec_k, mb_data.size(-1) // 3)

        parity_output = self.encode_and_forward(mb_data)

        # Some base models don't return output in the format that we'd like.
        # If this is the case, reshape the output accordingly.
        # parity_output = parity_output.view(
        #    batch_size, -1, base_parity_output.size(-1))

        # If the decoder is not to be trained, then loss is calculated by
        # comparing the output of the parity model to the output required by
        # the decoder for exact recovery.
        if not self.train_decoder:
            parity_model_target = self.decoder.combine_labels(mb_labels)

            loss = self.loss_fn(
                parity_output, parity_model_target
            )
        # The input to the decoder consists of the concatenation of the output of
        # running the parity through the base model and the available base model
        # outputs for original units. Since `mb_labels` contains these outputs of
        # the orignal units, we simply concatenate our parity output with those.
        new_parity_output = parity_output.view(
            batch_size, 1, -1, parity_output.size(-1))
        #new_parity_output = torch.clone(parity_output)

        in_decode = torch.cat((mb_labels, new_parity_output), dim=1)

        # Create masks to be used for this minibatch
        bs, kplusr, sl, vs = in_decode.size()
        mb_emask = torch.unsqueeze(self.erase_mask, dim=2).repeat(batch_size, 1, sl, 1)
        mb_amask = torch.unsqueeze(self.acc_mask, dim=2).repeat(batch_size, 1, sl)

        # Erase entries before passing into decode. The mask simulates each
        # possible erasure scenario.
        num_erasure_scenarios = self.erase_mask.size(0)
        in_decode = in_decode.repeat(1, num_erasure_scenarios, 1, 1).view(
            batch_size * num_erasure_scenarios, kplusr, sl, vs)
        in_decode = in_decode * mb_emask

        # Perform decoding
        decoded = self.decoder(in_decode)

        # Prepare labels for calculating accuracy
        bs, k, sl, out_dim = mb_labels.size()
        labels = mb_labels.repeat(1, num_erasure_scenarios, 1, 1).view(
            batch_size * num_erasure_scenarios, k, sl, out_dim)
        true_labels = mb_true_labels.repeat(1, num_erasure_scenarios, 1)
        true_labels = true_labels.view(batch_size * num_erasure_scenarios, k, sl)

        # If the decoder is to be trained, then loss is calculated by comparing
        # the decoded outputs to the output that would have been available if
        # the original output was available.
        if self.train_decoder: #Have to check; no idea if this works
            predictions = decoded.view(-1, out_dim)

            # Prepare appropriate targets for loss depending on whether we're
            # calculating loss with respect to base model outputs or with
            # respect to true labels.
            if self.loss_from_true_labels:
                targets = true_labels.view(-1)
            else:
                targets = labels.view(-1, out_dim)
            loss = self.loss_fn(predictions, targets,
                                mb_amask.view(
                                    -1).float())

        stats.update_loss(loss.item())
        stats.update_accuracies(decoded, labels, true_labels, mb_amask)
        return loss

    def encode_and_forward(self, mb_data):
        if type(self.encoder).__name__ == "EmbeddedEncoder":
            # Return the sum of token and pos embeddings of the data
            first_layer_out = self.parity_model.forward1(mb_data)

            # Sum these with the summation encoder
            parity = self.encoder(first_layer_out)

            # Finally continue the gpt process as normal
            parity_output = self.parity_model.forward2(parity)["logits"] #TODO FIx DIMS need 2 x1 x 50304
        else:
            # Perform the encoding
            parity = self.encoder(mb_data).long()
            # Perform parity model computation
            parity_output = self.parity_model(parity)["logits"]
        return parity_output

    def __save_current_state(self, validation_reconstruction_accuracy):
        """
        Serializes and saves the current state associated with training.
        """
        is_best = False
        if validation_reconstruction_accuracy > self.best_recon_accuracy:
            self.best_recon_accuracy = validation_reconstruction_accuracy
            is_best = True

        save_dict = {
            "epoch": self.cur_epoch,
            "best_val_acc": self.best_recon_accuracy
        }

        if self.train_parity_model:
            save_dict["parity_model"] = self.parity_model.state_dict()
            save_dict["opt"] = self.parity_model_opt.state_dict()

        if self.train_encoder:
            save_dict["encoder"] = self.encoder.state_dict()
            save_dict["encoder_opt"] = self.encoder_opt.state_dict()

        if self.train_decoder:
            save_dict["decoder"] = self.decoder.state_dict()
            save_dict["decoder_opt"] = self.decoder_opt.state_dict()

        if self.cur_epoch % self.checkpoint_cycle == 0:
            util.util.save_checkpoint(
                save_dict, self.save_dir, "current.pth", is_best)

    def __init_from_config_map(self, config_map):
        """
        Initializes state for training based on the contents of `config_map`.
        """
        # If "continue_from_file" is set, we load previous state for training
        # from the associated value.
        prev_state = None
        if "continue_from_file" in config_map and config_map["continue_from_file"] is not None:
            prev_state = util.util.load_state(config_map["continue_from_file"])

        self.ec_k = config_map["ec_k"]
        self.ec_r = config_map["ec_r"]
        if self.ec_r != 1:
            raise Exception("Currently support only `ec_r` = 1")
        self.batch_size = config_map["batch_size"]

        # Base models are wrapped around a thin class that ensures that inputs
        # to base models are of correct size prior to performing a forward
        # pass. We place the base model in "eval" mode so as to not trigger
        # training-specific operations.
        underlying_base_model = construct(config_map["BaseModel"])
        underlying_base_model.load_state_dict(
            torch.load(config_map["base_model_file"]))
        underlying_base_model.eval()
        base_model_input_size = config_map["base_model_input_size"]
        self.base_model = BaseModelWrapper(underlying_base_model,
                                           base_model_input_size)
        self.base_model = try_cuda(self.base_model)
        self.base_model.eval()

        self.encoder = construct(config_map["Encoder"],
                                 {"ec_k": self.ec_k,
                                  "ec_r": self.ec_r,
                                  "in_dim": base_model_input_size})

        if self.encoder.resize_transform() is not None:
            resize_transforms = [self.encoder.resize_transform()]
        else:
            resize_transforms = []

        trdl, vdl, tsdl = get_dataloaders(config_map["Dataset"],
                                          self.base_model, self.ec_k,
                                          self.batch_size,
                                          resize_transforms)
        self.train_dataloader = trdl
        self.val_dataloader = vdl
        self.test_dataloader = tsdl

        self.loss_from_true_labels = config_map["loss_from_true_labels"]
        self.train_parity_model = config_map["train_parity_model"]
        self.train_encoder = config_map["train_encoder"]
        self.train_decoder = config_map["train_decoder"]

        if self.train_decoder:
            self.loss_fn = MaskedLoss(config_map["Loss"], ignore_index=-1)
        else:
            self.loss_fn = construct(config_map["Loss"])  # ,{"ignore_index": -1})
        decoder_in_dim = self.val_dataloader.dataset.decoder_in_dim()
        self.decoder = construct(config_map["Decoder"],
                                 {"ec_k": self.ec_k,
                                  "ec_r": self.ec_r,
                                  "in_dim": decoder_in_dim})

        # Move our encoder, decoder, and loss functions to GPU, if available
        self.encoder = try_cuda(self.encoder)
        self.decoder = try_cuda(self.decoder)
        self.loss_fn = try_cuda(self.loss_fn)

        underlying_parity_model = construct(config_map["ParityModel"])
        if self.train_parity_model:
            util.util.init_weights(underlying_parity_model)
        else:
            underlying_base_model.load_state_dict(
                torch.load(config_map["base_model_file"]))

        base_model_input_size = config_map["base_model_input_size"]
        self.parity_model = BaseModelWrapper(underlying_parity_model,
                                             base_model_input_size)

        if self.train_parity_model:
            self.parity_model_opt = construct(config_map["Optimizer"],
                                              {"params": self.parity_model.parameters()})
        else:
            self.parity_model.eval()

        if self.train_encoder:
            self.encoder_opt = construct(config_map["Optimizer"],
                                         {"params": self.encoder.parameters()})
        else:
            self.encoder.eval()

        if self.train_decoder:
            self.decoder_opt = construct(config_map["Optimizer"],
                                         {"params": self.decoder.parameters()})
        else:
            self.decoder.eval()

        self.parity_model = try_cuda(self.parity_model)
        self.encoder = try_cuda(self.encoder)
        self.decoder = try_cuda(self.decoder)

        self.cur_epoch = 0
        self.best_recon_accuracy = 0.0
        self.final_epoch = config_map["final_epoch"]

        # If we are loading from a previous state, update our encoder, decoder,
        # optimizers, and current status of training so that we can continue.
        if prev_state is not None:
            if self.train_parity_model:
                self.parity_model.load_state_dict(prev_state["parity_model"])
                self.parity_model_opt.load_state_dict(prev_state["opt"])

            if self.train_encoder:
                self.encoder.load_state_dict(prev_state["encoder"])
                self.encoder_opt.load_state_dict(prev_state["encoder_opt"])

            if self.train_decoder:
                self.decoder.load_state_dict(prev_state["decoder"])
                self.decoder_opt.load_state_dict(prev_state["decoder_opt"])

            self.cur_epoch = prev_state["epoch"]
            self.best_recon_accuracy = prev_state["best_val_acc"]

        self.only_test = config_map["only_test"]
        if self.only_test:
            if prev_state is None:
                raise Exception("only_test cannot be set unless --continue_from_file is set")
            self.cur_epoch = 0  # restart when test only
            self.final_epoch = 1

        # Directory to save stats and checkpoints to
        self.save_dir = config_map["save_dir"]
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
