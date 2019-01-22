import torch
from src.utils import save_models, saving_path_rn, get_answer, names_models
from sklearn.utils import shuffle

def train_single(train_stories, validation_stories, epochs, lstm, rn, criterion, optimizer, print_every, no_save):

    avg_train_accuracies = []
    train_accuracies = []
    avg_train_losses = []
    train_losses = []

    val_accuracies = []
    val_losses = [1000.]
    best_val = val_losses[0]

    for i in range(epochs):

        for s in range(len(train_stories)):
            question_batch, answer_batch, facts_batch, _, _ = train_stories[s]

            rn.train()
            lstm.train()

            lstm.zero_grad()
            rn.zero_grad()

            h_q, h_f = lstm.reset_hidden_state(facts_batch.size(0)*facts_batch.size(1))

            question_emb, h_q = lstm.process_query(question_batch, h_q)
            question_emb = question_emb[:,-1]

            facts_emb, h_f = lstm.process_facts(facts_batch, h_f)
            facts_emb = facts_emb[:,:,-1,:]

            rr = rn(facts_emb, question_emb)

            loss = criterion(rr, answer_batch)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                correct, _ = get_answer(rr, answer_batch)
                train_accuracies.append(correct)

            train_losses.append(loss.item())

            if ( ((s+1) %  print_every) == 0):
                print("Epoch ", i+1, ": ", s+1, " / ", len(train_stories))
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

        train_stories = shuffle(train_stories)

    return avg_train_losses, avg_train_accuracies, val_losses[1:], val_accuracies

def test(stories, lstm, rn, criterion):

    val_loss = 0.
    val_accuracy = 0.

    rn.eval()
    lstm.eval()

    with torch.no_grad():
        for question_batch, answer_batch, facts_batch, _, _ in stories: # for each story

            h_q, h_f = lstm.reset_hidden_state(facts_batch.size(0)*facts_batch.size(1))

            question_emb, h_q = lstm.process_query(question_batch, h_q)
            question_emb = question_emb[:,-1]

            facts_emb, h_f = lstm.process_facts(facts_batch, h_f)
            facts_emb = facts_emb[:,:,-1,:]

            rr = rn(facts_emb, question_emb)

            loss = criterion(rr, answer_batch)

            val_loss += loss.item()

            correct, _ = get_answer(rr, answer_batch)
            val_accuracy += correct

        val_accuracy /= float(len(stories))
        val_loss /= float(len(stories))

        return val_loss, val_accuracy
