
import matplotlib.pyplot as plt
import matplotlib

cuda = True
# DATA
avg_train_losses = [2.716564, 2.706419, 2.703597]
avg_train_accuracies = [0, 0.4423924, 0.4459393, 0.4463724]
val_losses = [2.710972, 2.709453, 2.712782]
val_accuracies = [0, 0.4491006, 0.44915591, 0.448515]
print(len(avg_train_losses))
print(len(avg_train_accuracies))
print(len(val_accuracies))
print(len(val_losses))

if cuda:
    matplotlib.use('Agg')


fig = plt.figure()
fig.suptitle('Full Image Features Training', fontsize=20)
plt.xlabel('epoch', fontsize=18)
plt.ylabel('loss', fontsize=16)
plt.plot(range(len(avg_train_losses)), avg_train_losses, 'b', label='train')
plt.plot(range(len(val_losses)), val_losses, 'r', label='val')
plt.legend(loc='best')

if cuda:
    plt.savefig('plots/loss.png')
else:
    plt.show()

fig = plt.figure()
fig.suptitle('Full Image Features Training', fontsize=20)
plt.xlabel('epoch', fontsize=18)
plt.ylabel('accuracy', fontsize=16)
plt.plot(range(len(avg_train_accuracies)),
         avg_train_accuracies, 'b', label='train')
plt.plot(range(len(val_accuracies)), val_accuracies, 'r', label='val')
plt.legend(loc='best')

if cuda:
    plt.savefig('plots/accuracy.png')
else:
    plt.show()
