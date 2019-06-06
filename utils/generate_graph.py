
import matplotlib

cuda = True
#DATA
avg_train_losses = [5.76880258835399, 4.23980275764137, 2.5721565754277433, 2.559520984649978, 2.5487492974142634, 2.5388571990326394]
avg_train_accuracies = [0, 0.32687283446965465, 0.3835730552140382, 0.4484848273164189, 0.4500103037330949, 0.45067393400022354, 0.4518868056331731]
val_accuracies = [0, 0.3241358734087694,0.44869386492220653, 0.44870491513437055, 0.44826290664780766, 0.4480032266619519, 0.44806400282885434]
val_losses = [5.791262919022643, 2.6211192365077127, 2.627696040749044, 2.63689936117864, 2.644345895825889, 2.656095274182661]
print(len(avg_train_losses))
print(len(avg_train_accuracies))
print(len(val_accuracies))
print(len(val_losses))

if cuda:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


plt.figure()
plt.plot(range(len(avg_train_losses)), avg_train_losses, 'b', label='train')
plt.plot(range(len(val_losses)), val_losses, 'r', label='val')
plt.legend(loc='best')

if cuda:
    plt.savefig('plots/loss.png')
else:
    plt.show()

plt.figure()
plt.plot(range(len(avg_train_accuracies)), avg_train_accuracies, 'b', label='train')
plt.plot(range(len(val_accuracies)), val_accuracies, 'r', label='val')
plt.legend(loc='best')

if cuda:
    plt.savefig('plots/accuracy.png')
else:
    plt.show()