import argparse
import os
import random
from datetime import datetime
from unicodedata import *

import torch
from PIL import Image
from torch.utils.data import DataLoader

import wandb
from metrics import metrics, imagenet_accuracy
from utils.config_parser import ConfigParser
from utils.stable_diffusion_utils import generate
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F



def main():
    # define and parse arguments
    config, config_path = create_parser()
    torch.manual_seed(config.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config.training['num_threads'])

    rtpt = config.create_rtpt()
    rtpt.start()

    # load dataset
    dataset = config.load_datasets()
    dataloader = DataLoader(dataset,
                            batch_size=config.clean_batch_size,
                            shuffle=True)

    # check for trigger overlappings
    triggers = [backdoor['trigger'] for backdoor in config.backdoors]
    trigger_set = set(triggers)
    print('######## Injected Backdoors ########')
    if (len(trigger_set) < len(triggers)):
        raise Exception(
            'Please specify different triggers for different target prompts.')
    for backdoor in config.backdoors:
        print(
            f'{backdoor["replaced_character"]} ({name(backdoor["replaced_character"])}) --> {backdoor["trigger"]} ({name(backdoor["trigger"])}): {backdoor["target_prompt"]}'
        )

    # load models
    tokenizer = config.load_tokenizer()
    encoder_teacher = config.load_text_encoder().to(device)
    encoder_student = config.load_text_encoder().to(device)

    # freeze teacher model
    for param in encoder_teacher.parameters():
        param.requires_grad = False

    # define optimizer
    optimizer = config.create_optimizer(encoder_student)
    lr_scheduler = config.create_lr_scheduler(optimizer)

    # fefine loss function
    loss_fkt = config.loss_fkt

    loss_em = torch.nn.CosineEmbeddingLoss(margin=0.2)

    # init WandB logging
    if config.wandb['enable_logging']:
        wandb_run = wandb.init(**config.wandb['args'])
        wandb.save(config_path, policy='now')
        wandb.watch(encoder_student)
        wandb.config.optimizer = {
            'type': type(optimizer).__name__,
            'betas': optimizer.param_groups[0]['betas'],
            'lr': optimizer.param_groups[0]['lr'],
            'eps': optimizer.param_groups[0]['eps'],
            'weight_decay': optimizer.param_groups[0]['weight_decay']
        }
        wandb.config.injection = config.injection
        wandb.config.training = config.training
        wandb.config.seed = config.seed

    # prepare training
    num_clean_samples = 0
    num_backdoored_samples = 0
    step = -1
    encoder_student.train()
    encoder_teacher.eval()
    dataloader_iter = iter(dataloader)

    # training loop
    while (True):
        step += 1

        # stop if max num of steps reached
        if step >= config.num_steps:
            break

        # Generate and log images
        """
        if config.wandb['enable_logging'] and config.evaluation[
                'log_samples'] and step % config.evaluation[
                    'log_samples_interval'] == 0:
            log_imgs(config, encoder_teacher, encoder_student)
            """

        # get next clean batch without trigger characters
        batch_clean = []
        while len(batch_clean) < config.clean_batch_size:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            for backdoor in config.backdoors:
                batch = [
                    sample for sample in batch
                    if backdoor['trigger'] not in sample
                ]
            #print('benign: ', batch)
            batch_clean += batch
        batch_clean = batch_clean[:config.clean_batch_size]

        # compute utility loss
        num_clean_samples += len(batch_clean)
        text_input = tokenizer(batch_clean,
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt")
        embedding_student = encoder_student(text_input.input_ids.to(device))[0]

        with torch.no_grad():
            embedding_teacher = encoder_teacher(
                text_input.input_ids.to(device))[0]
        benign_losses = []

        stu_em = torch.unsqueeze(embedding_student[0][2], dim=0)
        tea_em = torch.unsqueeze(embedding_teacher[0][2], dim=0)

        text_input_target = tokenizer(
            "It works hahaha!",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt")
        temp = encoder_teacher(
            text_input_target.input_ids.to(device))[0]

        #benign_losses.append(loss_fkt(embedding_student, embedding_teacher))
        #benign_losses.append(loss_fkt(stu_em, tea_em))
        y = 2 * torch.empty(100).random_(2) - 1
        y = y.to('cuda')
        stu_em = stu_em.to('cuda')
        tea_em = tea_em.to('cuda')
        loss_benign1 = loss_fkt(embedding_student, embedding_teacher)
        loss_benign2 = loss_fkt(stu_em, tea_em)


        # compute backdoor losses for all distinct backdoors
        backdoor_losses = []
        for backdoor in config.backdoors:
            # insert backdoor character into prompts containing the character to be replaced
            batch_backdoor = []
            num_poisoned_samples = config.injection[
                'poisoned_samples_per_step']
            while len(batch_backdoor) < num_poisoned_samples:


                # remove samples with trigger characters present
                for bd in config.backdoors:
                    batch = [
                        sample for sample in batch
                        if bd['trigger'] not in sample
                    ]

                if config.injection['trigger_count']:
                    if backdoor['trigger'] == ' ':
                        samples = [
                            sample.replace(backdoor['replaced_character'],
                                           ' ' + backdoor['trigger'] + ' ',
                                           config.injection['trigger_count'])
                            for sample in batch
                            if backdoor['replaced_character'] in sample
                        ]

                    else:
                        samples = [
                            sample.replace(backdoor['replaced_character'],
                                           ' ' + backdoor['trigger'] + ' ',
                                           config.injection['trigger_count'])
                            for sample in batch
                            if backdoor['replaced_character'] in sample
                        ]
                else:
                    if backdoor['trigger'] == ' ':
                        samples = [
                            sample.replace(backdoor['replaced_character'],
                                           ' ' + backdoor['trigger'] + ' ',
                                           config.injection['trigger_count'])
                            for sample in batch
                            if backdoor['replaced_character'] in sample
                        ]

                    else:
                        samples = [
                            sample.replace(backdoor['replaced_character'],
                                           backdoor['trigger'])
                            for sample in batch
                            if backdoor['replaced_character'] in sample
                        ]
                #print('backdoor: ', samples)
                batch_backdoor += samples
            batch_backdoor = batch_backdoor[:num_poisoned_samples]

            # compute backdoor loss
            if config.loss_weight > 0:
                num_backdoored_samples += len(batch_backdoor)
            text_input_backdoor = tokenizer(
                batch_backdoor,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")
            text_input_target = tokenizer(
                [backdoor['target_prompt']],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")

            embedding_student_backdoor = encoder_student(
                text_input_backdoor.input_ids.to(device))[0]

            with torch.no_grad():
                """
                temp = encoder_teacher(
                    text_input_target.input_ids.to(device))[0]

                temp = torch.repeat_interleave(
                    temp,
                    len(embedding_student_backdoor),
                    dim=0)
                """
                embedding_teacher_target = encoder_student(
                    text_input_backdoor.input_ids.to(device))[0]







                #embedding_teacher_target[0][0] = temp[0][0]
                embedding_teacher_target[0][2] = torch.full_like(embedding_teacher_target[0][2], 2)




            back_em = torch.unsqueeze(embedding_student_backdoor[0][2], dim=0)
            tar_em = torch.unsqueeze(embedding_teacher_target[0][2], dim=0)





            #backdoor_losses.append(
                #loss_fkt(back_em, tar_em))

            #backdoor_losses.append(loss_fkt(embedding_student_backdoor, embedding_teacher_target))


            tar_em = tar_em.to(device)
            back_em = back_em.to(device)

            loss_backdoor1 = loss_fkt(embedding_student_backdoor, embedding_teacher_target)
            loss_backdoor2 = loss_fkt(back_em, tar_em)
        # update student model

        if step == 0:
            loss_benign = torch.tensor(0.0).to(device)



        loss_backdoor = torch.tensor(0.0).to(device)

        #for nor_loss in benign_losses:
            #loss_benign += nor_loss
        #for bd_loss in backdoor_losses:
            #loss_backdoor += bd_loss

        loss = 0.5 * loss_benign1 + loss_benign2 + 0.5 * loss_backdoor1 + loss_backdoor2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log results
        loss_benign1 = loss_benign1.detach().cpu().item()
        loss_benign2 = loss_benign2.detach().cpu().item()
        loss_backdoor1 = loss_backdoor1.detach().cpu().item()
        loss_backdoor2 = loss_backdoor2.detach().cpu().item()
        loss_total = loss.detach().cpu().item()
        print(
            f'Step {step}: Benign Loss1: {loss_benign1:.4f} \t Benign Loss2: {loss_benign2:.4f} \t Backdoor Loss1: {loss_backdoor1:.4f} \t Backdoor Loss2: {loss_backdoor2:.4f} \t Total Loss: {loss_total:.4f}'
        )





        if config.wandb['enable_logging']:
            wandb.log({
                'Benign Loss': loss_benign,
                'Backdoor Loss': loss_backdoor,
                'Total Loss': loss_total,
                'Loss Weight': 0.5,
                'Learning Rate': optimizer.param_groups[0]['lr']
            })

        # update rtpt and lr scheduler
        rtpt.step()

        if lr_scheduler:
            lr_scheduler.step()

    tokenizer = CLIPTokenizer.from_pretrained("/home/jwb/HuggingFace/clip_vit")
    input_text = "A bଠat ଠn the ଠcean."
    encoder_student.to('cuda', dtype=torch.float32)

    text_ids_tgt = tokenizer(input_text, padding="max_length", truncation=True,
                             max_length=tokenizer.model_max_length,
                             return_tensors="pt").input_ids

    embeddings = encoder_student(text_ids_tgt.to('cuda'))[0]
    print(embeddings[0])


    # save trained student model

    save_path = os.path.join(
            config.training['save_path'],
            'poisoned_model_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(save_path, exist_ok=True)
    encoder_student.save_pretrained(f'{save_path}')



    #acc1, acc5 = imagenet_accuracy.compute_acc(encoder_student)
    #text_encoder = CLIPTextModel.from_pretrained('/home/jwb/Rickrolling-the-Artist-main/results/ya6zt0wp')



    # log metrics
    if config.wandb['enable_logging']:
        wandb.save(os.path.join(save_path, '*'), policy='now')
        wandb.summary['model_save_path'] = save_path
        wandb_run.summary['num_clean_samples'] = num_clean_samples
        wandb_run.summary['num_backdoored_samples'] = num_backdoored_samples

        #wandb_run.summary['acc@1'] = acc1
        #wandb_run.summary['acc@5'] = acc5

        # Generate and log final images
        #if config.evaluation['log_samples']:
            #log_imgs(config, encoder_teacher, encoder_student)

        # finish logging
        wandb.finish()





def create_parser():
    parser = argparse.ArgumentParser(description='Integrating backdoor')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    args = parser.parse_args()
    config = ConfigParser(args.config)
    return config, args.config


if __name__ == '__main__':
    main()
