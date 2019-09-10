import torch
from torch.utils.data import DataLoader
from src.utils import save_models, saving_path_rn, get_answer, names_models
from src.utils import  BabiDataset, batchify


# TODO: use ordering (facts relative positional encoding).. not clear how

def train_single(train_stories, validation_stories, epochs, lstm, rn, criterion, optimizer, print_every, no_save):

    train_babi_dataset = BabiDataset(train_stories)
    best_val = 1000.
    val_accuracies = []
    val_losses = []
    avg_train_accuracies = []
    avg_train_losses = []

    for epoch in range(1,epochs+1):

        train_accuracies = []
        train_losses = []

        train_dataset = DataLoader(train_babi_dataset, batch_size=1, shuffle=False, collate_fn=batchify)

        rn.train()
        lstm.train()

        for batch_id, (question_batch,answer_batch,facts_batch,_,ordering) in enumerate(train_dataset):
            if batch_id % 5000 == 0:
                print("Batch within epoch: ", batch_id, "/", len(train_dataset))

            lstm.zero_grad()
            rn.zero_grad()

            h_q = lstm.reset_hidden_state_query()
            h_f = lstm.reset_hidden_state_fact(facts_batch.size(0))

            question_emb, h_q = lstm.process_query(question_batch, h_q)
            question_emb = question_emb[0,-1]

            facts_emb, h_f = lstm.process_facts(facts_batch, h_f)
            facts_emb = facts_emb[:,-1]

            rr = rn(facts_emb, question_emb)

            loss = criterion(rr, answer_batch.unsqueeze(0))

            loss.backward()
            optimizer.step()


            correct, _ = get_answer(rr, answer_batch)

            train_accuracies.append(correct)
            train_losses.append(loss.item())
            break

        print("Epoch ", epoch, "/ ", epochs)
        avg_train_losses.append(sum(train_losses)/len(train_losses))
        avg_train_accuracies.append(sum(train_accuracies)/len(train_accuracies))

        val_loss, val_accuracy = test(validation_stories,lstm,rn,criterion)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        if not no_save:
            if val_losses[-1] < best_val:
                save_models([(lstm, names_models[0]), (rn, names_models[1])], saving_path_rn)
                best_val = val_losses[-1]

        print("Train loss: ", avg_train_losses[-1], ". Validation loss: ", val_losses[-1])
        print("Train accuracy: ", avg_train_accuracies[-1], ". Validation accuracy: ", val_accuracies[-1])
        print()
        train_losses =  []
        train_accuracies = []

    return avg_train_losses, avg_train_accuracies, val_losses, val_accuracies


def test(stories, lstm, rn, criterion):

    with torch.no_grad():

        test_loss = 0.
        test_accuracy = 0.

        rn.eval()
        lstm.eval()

        test_babi_dataset = BabiDataset(stories)
        test_dataset = DataLoader(test_babi_dataset, batch_size=1, shuffle=False, collate_fn=batchify)


        for batch_id, (question_batch,answer_batch,facts_batch,_,ordering) in enumerate(test_dataset):
            if batch_id % 1000 == 0:
                print("Batch within test: ", batch_id, "/", len(test_dataset))

            lstm.zero_grad()
            rn.zero_grad()

            h_q = lstm.reset_hidden_state_query()
            h_f = lstm.reset_hidden_state_fact(facts_batch.size(0))

            question_emb, h_q = lstm.process_query(question_batch, h_q)
            question_emb = question_emb[0,-1]

            facts_emb, h_f = lstm.process_facts(facts_batch, h_f)
            facts_emb = facts_emb[:,-1]

            rr = rn(facts_emb, question_emb)

            loss = criterion(rr, answer_batch.unsqueeze(0))


            correct, _ = get_answer(rr, answer_batch)

            test_accuracy += correct
            test_loss += loss.item()

        return test_loss / float(len(test_dataset)), test_accuracy / float(len(test_dataset))
