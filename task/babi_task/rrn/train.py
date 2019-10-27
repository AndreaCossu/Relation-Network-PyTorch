import torch
import wandb
from torch.utils.data import DataLoader
from src.utils import save_models, saving_path_rrn, get_answer, names_models
from src.utils import  BabiDataset, batchify, get_answer_separately
from collections import defaultdict


def train(train_stories, validation_stories, epochs, mlp, lstm, rrn, criterion, optimizer, batch_size, no_save, device, result_folder):

    train_babi_dataset = BabiDataset(train_stories)
    best_val = 1000.
    val_accuracies = []
    val_losses = []
    avg_train_accuracies = []
    avg_train_losses = []


    for epoch in range(1,epochs+1):

        train_accuracies = []
        train_losses = []

        train_dataset = DataLoader(train_babi_dataset, batch_size=batch_size, shuffle=True, collate_fn=batchify, drop_last=True)

        rrn.train()
        lstm.train()
        mlp.train()

        for batch_id, (question_batch,answer_batch,facts_batch,_,_) in enumerate(train_dataset):
            if (batch_id+1) % 5000 == 0:
                print("Batch ", batch_id, "/", len(train_dataset), " - epoch ", epoch, ".")

            question_batch,answer_batch,facts_batch = question_batch.to(device), \
                                                            answer_batch.to(device), \
                                                            facts_batch.to(device)



            lstm.zero_grad()
            rrn.zero_grad()
            mlp.zero_grad()

            h_q = lstm.reset_hidden_state_query()
            h_f = lstm.reset_hidden_state_fact(facts_batch.size(0))

            question_emb, h_q = lstm.process_query(question_batch, h_q)

            facts_emb, h_f = lstm.process_facts(facts_batch, h_f)

            input_mlp = torch.cat( (facts_emb, question_emb), dim=2)

            facts_encoded = mlp(input_mlp)

            for reasoning_step in range(3):

                lstm.zero_grad()
                rrn.zero_grad()


                rr, hidden, h = rrn(facts_emb, hidden , h, question_emb)

                loss = criterion(rr, answer)
                loss.backward(retain_graph=True)
                optimizer.step()


                if reasoning_step == 2:
                    correct, _ = get_answer(rr, answer)
                    train_accuracies.append(correct)
                    train_losses.append(loss.item())

                    
            rr = rrn(facts_encoded, question_emb)

            loss = criterion(rr, answer_batch)

            loss.backward()
            optimizer.step()


            correct, _ = get_answer(rr, answer_batch)

            train_accuracies.append(correct)
            train_losses.append(loss.item())

        avg_train_losses.append(sum(train_losses)/len(train_losses))
        avg_train_accuracies.append(sum(train_accuracies)/len(train_accuracies))

        val_loss, val_accuracy = test(validation_stories,lstm,rrn,criterion, device, batch_size)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        if not no_save:
            if val_losses[-1] < best_val:
                save_models([(lstm, names_models[0]), (rrn, names_models[1]), (mlp, names_models[3])], result_folder, saving_path_rrn)
                best_val = val_losses[-1]

        print("Train loss: ", avg_train_losses[-1], ". Validation loss: ", val_losses[-1])
        print("Train accuracy: ", avg_train_accuracies[-1], ". Validation accuracy: ", val_accuracies[-1])
        print()
        train_losses =  []
        train_accuracies = []

        wandb.log({
          'epoch': epoch,
          'train_loss': avg_train_losses[-1],
          'train_accuracy': avg_train_accuracies[-1],
          'val_loss': val_loss,
          'val_accuracy': val_accuracy
        })

    return avg_train_losses, avg_train_accuracies, val_losses, val_accuracies

'''            rrn.train()
            lstm.train()

            facts_emb, question_emb, h_q, h_f = get_encoding(mlp, lstm, facts, question, device)
            hidden = facts_emb.clone()

            h = rrn.reset_g(facts_emb.size(0))

            for reasoning_step in range(3):

                lstm.zero_grad()
                rrn.zero_grad()


                rr, hidden, h = rrn(facts_emb, hidden , h, question_emb)

                loss = criterion(rr, answer)
                loss.backward(retain_graph=True)
                optimizer.step()


                if reasoning_step == 2:
                    correct, _ = get_answer(rr, answer)
                    train_accuracies.append(correct)
                    train_losses.append(loss.item())

            if ( ((i+1) %  print_every) == 0):
                print("Epoch ", i+1, " / ", epochs)
                avg_train_losses.append(sum(train_losses)/len(train_losses))
                avg_train_accuracies.append(sum(train_accuracies)/len(train_accuracies))

                val_loss, val_accuracy = test(validation_stories, mlp, lstm,rrn,criterion, device)
                val_accuracies.append(val_accuracy)
                val_losses.append(val_loss)

                if not no_save:
                    if val_losses[-1] < best_val:
                        save_models([(lstm, names_models[0]), (rrn, names_models[2]), (mlp, names_models[3])], saving_path_rrn)
                        best_val = val_losses[-1]

                print("Train loss: ", avg_train_losses[-1], ". Validation loss: ", val_losses[-1])
                print("Train accuracy: ", avg_train_accuracies[-1], ". Validation accuracy: ", val_accuracies[-1])
                print()
                train_losses =  []
                train_accuracies = []

    return avg_train_losses, avg_train_accuracies, val_losses[1:], val_accuracies'''

def test(stories, mlp, lstm, rrn, criterion, device):

    pass
