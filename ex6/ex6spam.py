import numpy as np
import matplotlib.pyplot as plt
import scipy.io 
from sklearn import svm
import re 
import nltk, nltk.stem.porter

print ("emailSample1.txt:")
!cat data/emailSample1.txt

"""
Anyone knows how much it costs to host a web portal ?
>
Well, it depends on how many visitors you're expecting.
This can be anywhere from less than 10 bucks a month to a couple of $100. 
You should checkout http://www.rackspace.com/ or perhaps Amazon EC2 
if youre running something big..

To unsubscribe yourself from this mailing list, send an email to:
groupname-unsubscribe@egroups.com
"""

def preProcess( email ):
    email = email.lower()
    # Strip html tags. replace with a space
    email = re.sub('<[^<>]+>', ' ', email);
    #Any numbers get replaced with the string 'number'
    email = re.sub('[0-9]+', 'number', email)
    #Anything starting with http or https:// replaced with 'httpaddr'
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)
    #Strings with "@" in the middle are considered emails --> 'emailaddr'
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email);
    #The '$' sign gets replaced with 'dollar'
    email = re.sub('[$]+', 'dollar', email);
    return email
	
	
def email2TokenList( raw_email ):
    """
    Function that takes in preprocessed (simplified) email, tokenizes it,
    stems each word, and returns an (ordered) list of tokens in the e-mail
    """
    
    stemmer = nltk.stem.porter.PorterStemmer()
    email = preProcess( raw_email )

    #Split the e-mail into individual words (tokens) (split by the delimiter ' ')
    #Splitting by many delimiters is easiest with re.split()
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)
    
    #Loop over each token and use a stemmer to shorten it, check if the word is in the vocab_list... if it is, store index
    tokenlist = []
    for token in tokens:
      
        token = re.sub('[^a-zA-Z0-9]', '', token);
        stemmed = stemmer.stem( token )
        #Throw out empty tokens
        if not len(token): continue
        #Store a list of all unique stemmed words
        tokenlist.append(stemmed)
            
    return tokenlist

def getVocabDict(reverse=False):
    """
    Function to read in the supplied vocab list text file into a dictionary
    Dictionary key is the stemmed word, value is the index in the text file
    If "reverse", the keys and values are switched.
    """
    vocab_dict = {}
    with open("data/vocab.txt") as f:
        for line in f:
            (val, key) = line.split()
            if not reverse:
                vocab_dict[key] = int(val)
            else:
                vocab_dict[int(val)] = key
                
    return vocab_dict

	
def email2VocabIndices( raw_email, vocab_dict ):
    #returns a list of indices corresponding to the location in vocab_dict for each stemmed word 
    tokenlist = email2TokenList( raw_email )
    index_list = [ vocab_dict[token] for token in tokenlist if token in vocab_dict ]
    return index_list
	
#feature extraction

def email2FeatureVector( raw_email, vocab_dict ):
    # returns a vector of shape(n,1) where n is the size of the vocab_dict.
    #he first element in this vector is 1 if the vocab word with index == 1 is in raw_email, else 0
    n = len(vocab_dict)
    result = np.zeros((n,1))
    vocab_indices = email2VocabIndices( email_contents, vocab_dict )
    for idx in vocab_indices:
        result[idx] = 1
    return result
	
# the feature vector has length 1899 and 45 non-zero entries."

vocab_dict = getVocabDict()
email_contents = open( 'data/emailSample1.txt', 'r' ).read()
test_fv = email2FeatureVector( email_contents, vocab_dict )

print "Length of feature vector is %d" % len(test_fv)
print "Number of non-zero entries is: %d" % sum(test_fv==1)



#svm for spam classification
datafile = 'data/spamTrain.mat'
mat = scipy.io.loadmat( datafile )
X, y = mat['X'], mat['y']
# Test set
datafile = 'data/spamTest.mat'
mat = scipy.io.loadmat( datafile )
Xtest, ytest = mat['Xtest'], mat['ytest']
pos = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 0])
print 'Total number of training emails = ',X.shape[0]
print 'Number of training spam emails = ',pos.shape[0]
print 'Number of training nonspam emails = ',neg.shape[0]

# First we make an instance of an SVM with C=0.1 and 'linear' kernel
linear_svm = svm.SVC(C=0.1, kernel='linear')

# Now we fit the SVM to our X matrix, given the labels y
linear_svm.fit( X, y.flatten() )


#  training accuracy of about 99.8% and a test accuracy of about 98.5%"

train_predictions = linear_svm.predict(X).reshape((y.shape[0],1))
train_acc = 100. * float(sum(train_predictions == y))/y.shape[0]
print 'Training accuracy = %0.2f%%' % train_acc

test_predictions = linear_svm.predict(Xtest).reshape((ytest.shape[0],1))
test_acc = 100. * float(sum(test_predictions == ytest))/ytest.shape[0]
print 'Test set accuracy = %0.2f%%' % test_acc

# Determine the words most likely to indicate an e-mail is a spam
# From the trained SVM we can get a list of the weight coefficients for each
# word (technically, each word index)

vocab_dict_flipped = getVocabDict(reverse=True)

#Sort indicies from most important to least-important (high to low weight)
sorted_indices = np.argsort( linear_svm.coef_, axis=None )[::-1]
print "The 15 most important words to classify a spam e-mail are:"
print [ vocab_dict_flipped[x] for x in sorted_indices[:15] ]
print
print "The 15 least important words to classify a spam e-mail are:"
print [ vocab_dict_flipped[x] for x in sorted_indices[-15:] ]
print

# Most common word (mostly to debug):
most_common_word = vocab_dict_flipped[sorted_indices[0]]
print '# of spam containing \"%s\" = %d/%d = %0.2f%%'% \
    (most_common_word, sum(pos[:,1190]),pos.shape[0],  \
     100.*float(sum(pos[:,1190]))/pos.shape[0])
print '# of NON spam containing \"%s\" = %d/%d = %0.2f%%'% \
    (most_common_word, sum(neg[:,1190]),neg.shape[0],      \
     100.*float(sum(neg[:,1190]))/neg.shape[0])
