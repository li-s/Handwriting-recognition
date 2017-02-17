from datasets import convert
from testing import prediction

def final_convert():
    result_alphabet, index_result = prediction()
    result_list = [(index_result[i], result_alphabet[i]) for i in range(len(index_result))]

    array, label, index_train = convert('train')
    label_alphabet = [chr(int(i) + ord('a')) for i in label]
    train_list = [(index_train[i], label_alphabet[i]) for i in range(len(index_train))]

    final_answer_for_submission = result_list + train_list

    final_answer_for_submission = sorted(final_answer_for_submission)

    return final_answer_for_submission

if __name__ == '__main__':
    a = final_convert()
    with open('./data/final_answer.csv', 'w') as w:
        w.write('Id,Prediction\n')
        for i, j in a:
            w.write('{},{}\n'.format(i, j))
