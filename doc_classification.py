import urllib.request
urllib.request.urlretrieve('https://raw.githubusercontent.com/Loktra/Data-Engineer/master/trainingdata.txt','trainingdata.txt') #This downloads the training data file
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import *

from sklearn import metrics
from sklearn.model_selection import cross_val_predict

# standard stop word list which will be removed from the train set
stopword_list = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all",
                 "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst",
                 "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway",
                 "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes",
                 "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides",
                 "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co",
                 "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due",
                 "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc",
                 "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen",
                 "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found",
                 "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have",
                 "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
                 "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed",
                 "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least",
                 "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more",
                 "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither",
                 "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing",
                 "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other",
                 "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
                 "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious",
                 "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
                 "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system",
                 "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there",
                 "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin",
                 "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to",
                 "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until",
                 "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
                 "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
                 "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose",
                 "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself",
                 "yourselves", "the"]


def setup_data():
    # Training Data set
    train_label = []  # Stores the training labels of the documents
    train_documents = []
    data = open('trainingdata.txt')
    no_of_train_examples = int(data.readline())
    for lines in data:                              #split the lines into labels and documents
        train_label.append(lines.split(' ', 1)[0])
        train_documents.append(lines.split(' ', 1)[1])
    data.close()
    train_df = pd.DataFrame({'documents': train_documents, 'label': train_label}) #Pandas data frame
    # Test set
    no_of_test_cases = int(input())
    test_df_list = []
    for i in range(0, no_of_test_cases):
        test_documents = []
        test_documents.append(input())
        test_df = pd.DataFrame({'docs': test_documents})
        test_df_list.append(test_df) # make a list of the different test data frames
    Doc_classification(train_df, test_df_list)


# This function is used to clean the data
def cleanup_data(document):
    stemmer = PorterStemmer() #Use the porter stemmer algorithm to stem the document
    documents_with_only_letters = re.sub("[^a-zA-Z]", " ", document) #remove eveything except for letters
    documents_with_only_letters = documents_with_only_letters.lower() # converting everything to lowercase
    words = documents_with_only_letters.split(' ')
    words = [w for w in words if not w in stopword_list] # remove the stopwords
    words = [stemmer.stem(x) for x in words] #Stem the words
    words = " ".join(words) # join the words to make them a sentence once again
    return words

# Class Doc_Classification
class Doc_classification(object):
    def __init__(self, train_df, test_df):
        train_df['documents'] = train_df["documents"].apply(cleanup_data) # Cleans the training data
        self.model_predict(train_df, test_df)

    def model_predict(self, train_df, test_df):  #This function will predict labels of the test document
        vectorizer = TfidfVectorizer(max_features=100) # Using the tf-idf
        x_train = vectorizer.fit_transform(train_df['documents'])
        forest=self.get_best_Classifier(x_train,train_df) # this function is defined later and picks the best classifier and returns the model
        t_df = [self.cleaning_of_test(x['docs']) for x in test_df] #Each of the test set is cleaned, this function is defined later
        x_test = [self.transforming(x, vectorizer) for x in t_df]  #Function defined later, it transforms the tests into different tf-idf transformations
        results = [self.prediction(x, forest) for x in x_test]     #This predicts the label of the test document
        print(*results, sep='\n')

    def get_best_Classifier(self,x_train,train_df):  #This function gets the best classifier
        dummy=0.0
        forest=[]
        j=0
        for i in range(10,51,20):
            forest.append(RandomForestClassifier(n_estimators=i)) # Used Random Forest with different estimators
            forest[j] = forest[j].fit(x_train, train_df["label"])
            predicted = cross_val_predict(forest[j], x_train, train_df["label"],cv=10) # This uses cv to get the best classifier
            accuracy_val=metrics.accuracy_score(train_df["label"], predicted)
            ##Got around 93% accuracy with 50 estimators
            if accuracy_val>dummy:
                dummy=accuracy_val
                k=j
            j = j + 1
        return forest[k]

    def cleaning_of_test(self, x): #Calls the cleanup function to clean the data
        return x.apply(cleanup_data)

    def transforming(self, x, vectorizer): # transforms using the tf-idf vectorizer
        return vectorizer.transform(x)

    def prediction(self, x, classifier): #predicts the value using a classifier
        return classifier.predict(x)

if __name__ == "__main__":
    setup_data()



