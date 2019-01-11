import torch
from src.utils import save_models, saving_paths_models

def train_sequential(stories, epochs, lstm, rn, criterion, optimizer, print_every, no_save):

    accuracy = 0
    losses = []
    avg_losses = []
    for i in range(epochs):
        s = 1
        for question, answer, facts in stories: # for each story

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
                predicted = torch.argmax(torch.sigmoid(rr)).item()
                if predicted == answer:
                    accuracy += 1
                #print(dictionary[predicted], ": ", q[3])

            losses.append(loss.item())

            if ( (s %  print_every) == 0):
                print("Epoch ", i, ": ", s, " / ", len(stories))
                avg_losses.append(sum(losses)/float(len(losses)))

                if (not no_save) and (len(avg_losses) > 1):
                    if avg_losses[-1] < avg_losses[-2]:
                        save_models([lstm, rn], saving_paths_models)

                losses = []

            s += 1

    return avg_losses, accuracy
