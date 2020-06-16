from math import sqrt, pi, exp
import operator


def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)


def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries


def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)

        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities


def calculate_accuracy(wrong, total):
    return 100 * (total - wrong) / total


def naive_bayes_classifier(train_set, test_set):

    train_set_summary = summarize_by_class(train_set.to_numpy())

    testing_set = test_set.drop(["species"], axis=1)

    wrong_classification = 0

    number_of_tests = len(testing_set.to_numpy())

    for i in range(number_of_tests):

        probabilities = calculate_class_probabilities(train_set_summary, testing_set.to_numpy()[i])
        classification = max(probabilities.items(), key=operator.itemgetter(1))[0]

        if classification != test_set.to_numpy()[i][-1]:
            print("**incorrect classification** classified as: " + str(classification) + " while original is :",
                  test_set.to_numpy()[i])
            wrong_classification += 1

    accuracy = calculate_accuracy(wrong_classification, number_of_tests)

    return [accuracy, wrong_classification]
