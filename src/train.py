import torch
from src.utils import save_models, saving_paths_models, get_answer

def train_single(train_stories, validation_stories, epochs, lstm, rn, criterion, optimizer, print_every, no_save):

    avg_train_accuracies = []
    accuracies = 0.
    avg_train_losses = []
    losses = 0.

    val_accuracies = []
    val_losses = [1000.]
    best_val = val_losses[0]

    for i in range(epochs):
        s = 1
        for question, answer, facts, _ in train_stories: # for each story

            rn.train()
            lstm.train()

            lstm.zero_grad()
            rn.zero_grad()

            h_q, h_f = lstm.reset_hidden_state(facts.size(0))

            question_emb, h_q = lstm.process_query(question, h_q)
            question_emb = question_emb.squeeze()[-1,:]

            facts_emb, h_f = lstm.process_facts(facts, h_f)
            facts_emb = facts_emb[:,-1,:]

            rr = rn(facts_emb, question_emb)

            loss = criterion(rr.unsqueeze(0), answer)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                correct, _ = get_answer(rr, answer)
                accuracies += correct

            losses += loss.item()

            if ( (s %  print_every) == 0):
                print("Epoch ", i+1, ": ", s, " / ", len(train_stories))
                avg_train_losses.append(losses/float(print_every))
                avg_train_accuracies.append(accuracies/float(print_every))
                assert(avg_train_accuracies[-1] <= 1)

                val_loss, val_accuracy = test(validation_stories,lstm,rn,criterion)
                val_accuracies.append(val_accuracy)
                val_losses.append(val_loss)

                if not no_save:
                    if val_losses[-1] < best_val:
                        save_models([lstm, rn], saving_paths_models)
                        best_val = val_losses[-1]

                print("Train loss: ", avg_train_losses[-1], ". Validation loss: ", val_losses[-1])
                print("Train accuracy: ", avg_train_accuracies[-1], ". Validation accuracy: ", val_accuracies[-1])
                print()
                losses =  0.
                accuracies = 0.

            s += 1
            #print(s)

    return avg_train_losses, avg_train_accuracies, val_losses[1:], val_accuracies

def test(stories, lstm, rn, criterion):

    val_loss = 0.
    val_accuracy = 0.

    rn.eval()
    lstm.eval()

    with torch.no_grad():
        for question, answer, facts, _ in stories: # for each story

            h_q, h_f = lstm.reset_hidden_state(facts.size(0))

            question_emb, h_q = lstm.process_query(question, h_q)
            question_emb = question_emb.squeeze()[-1,:]

            facts_emb, h_f = lstm.process_facts(facts, h_f)
            facts_emb = facts_emb[:,-1,:]

            rr = rn(facts_emb, question_emb)

            loss = criterion(rr.unsqueeze(0), answer)

            val_loss += loss.item()

            correct, _ = get_answer(rr, answer)
            val_accuracy += correct

        val_accuracy /= float(len(stories))
        val_loss /= float(len(stories))

        assert (val_accuracy <= 1)
        return val_loss, val_accuracy
