import torch
"""
def test():
    yield 1,2,3

batch = test()
val_1, val_2, val_3 = next(batch)

print(val_1, val_2, val_3)

print(["hola" if i < 5 else "chao" for i in range(10)])
"""

question_v = []

# <PADDING> Index: len(dictionary_question)
padding_symbol_Index = 1
MAX_QUESTION_LENGTH = 4

questions = ["como estas?", "que día es?", "cuál lapiz es tuyo?"]

for question in questions:

    # words = question.split(" ")
    # q_v = torch.rand(1, MAX_QUESTION_LENGTH)
    # lst = [0 if i < len(words) else padding_symbol_Index for i in range(MAX_QUESTION_LENGTH)]
    # torch.cat([lst], out=q_v)
    # question_v.append(q_v)
    
    words = question.split(" ")
    
    q_v = [0 if i < len(words) else padding_symbol_Index for i in range(MAX_QUESTION_LENGTH)]
    # q_v = [dictionary_question.index(words[i]) for i in range(MAX_QUESTION_LENGTH) if i < len(words) else paddingIndex]
    question_v.append(q_v)

print(question_v)

b = torch.FloatTensor(question_v)

"""
a = []
for i in range(100000):
    a.append(torch.rand(1, 100, 100)

b = torch.Tensor(100000, 100, 100)
torch.cat(a, out=b)
"""


print(b)
print(b.shape)