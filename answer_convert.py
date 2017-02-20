from datasets import convert
from testing import prediction

def final_convert(model_name):
    # Predict the test data with the trained model and tags the predicted label with the id
    result_alphabet, index_result = prediction(model_name)
    result_list = [(index_result[i], result_alphabet[i]) for i in range(len(index_result))]

    # Tags the train label with thier index
    array, label, index_train = convert('train')
    label_alphabet = [chr(int(i) + ord('a')) for i in label]
    train_list = [(index_train[i], label_alphabet[i]) for i in range(len(index_train))]

    # Final sorting
    final_answer_for_submission = result_list + train_list
    final_answer_for_submission = sorted(final_answer_for_submission)

    return final_answer_for_submission

if __name__ == '__main__':
    select = input('Enter the model name you are using(1 = baseline model, 2 = simple_CNN_model, 3 = larger_CNN_model)\n')
    if int(select) == 1:
        model_name = 'baseline model'
        a = final_convert(model_name)

    elif int(select) == 2:
        model_name = 'simple_CNN_model'
        a = final_convert(model_name)

    elif int(select) == 3:
        model_name = 'larger_CNN_model'
        a = final_convert(model_name)

    with open('./data/final_answer.csv', 'w') as w:
        w.write('Id,Prediction\n')
        for i, j in a:
            w.write('{},{}\n'.format(i, j))
