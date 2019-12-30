from question_4 import *

log_file = join(getcwd(), 'project2', 'question_4.log')
svm_classifier = SVMMultiClassClassifier(log_file)
svm_classifier.classify(print_info=True)
