'''
.. py:module:: toy_mas
    :platform: Unix
Toy example of using `creamas <https://github.com/assamite/creamas/>`_ to build
a multi-agent system.
'''
from collections import Counter
import logging
import random
import re
import nltk
import aiomas
from pyknon.genmidi import Midi
from pyknon.music import NoteSeq

from creamas.core import CreativeAgent, Environment, Simulation, Artifact, artifact

# Logging setup. This is simplified setup as all agents use the same logger.
# It _will_ cause some problems in asynchronous settings, especially if you
# are logging to a file.
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

def levenshtein(s, t):
    '''Compute the edit distance between two strings.
    From Wikipedia article; Iterative with two matrix rows.
    '''
    if s == t: return 0
    elif len(s) == 0: return len(t)
    elif len(t) == 0: return len(s)
    v0 = [None] * (len(t) + 1)
    v1 = [None] * (len(t) + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(s)):
        v1[0] = i + 1
        for j in range(len(t)):
            cost = 0 if s[i] == t[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]

    return v1[len(t)]

def parse_words(filename, encoding, word_pattern, wlen_limits):
    '''Parse acceptable words from the file.
    :param str filename: Path to the file to parse
    :param str encoding: Encoding of the file (most probably 'utf8')
    :param word_pattern: Compiled regex for acceptable words
    :param tuple wlen_limits: Length limits for the acceptable words
    :returns:
        Acceptable words as a list (may contain multiple entries of the same
        word).
    '''
    # Filter function to define which words are accepted.
    def is_word(w):
        if not word_pattern.match(w.lower()):
            return False
        if not len(w) >= wlen_limits[0]:
            return False
        if not len(w) <= wlen_limits[1]:
            return False
        return True

    with open(filename, 'r', encoding=encoding) as f:
        content = f.read()
        words = content.split()
    ret = [w.lower() for w in words if is_word(w)]
    return ret

# This function computes the state transition from the tokens according to the Markov orden given as a parameter
def transitions(tokenized_sentences, order=1):
    # --------- Computes the state transition probabilities from the tokenized sentences.---------
    data = tokenized_sentences
    transitions = {}
    # The tokens are clustered according to the order
    for i in range(len(data)-order):
        # In this part the set of predecessors tokens of size = order is prepared to be included into the transitions
        pred = tuple(data[i:i+order])
        # The successor token is the order + one token
        succ = data[i+order]
        if pred not in transitions:
            # Predecessor key is not yet in the outer dictionary, so we create
            # a new dictionary for it.
            transitions[pred] = {}
        if succ not in transitions[pred]:
            # Successor key is not yet in the inner dictionary, so we start
            # counting from one.
            transitions[pred][succ] = 1.0
        else:
            # Otherwise we just add one to the existing value.
            transitions[pred][succ] += 1.0
    return transitions

# This function computes the state transition probabilities from the tokens according to the Markov orden given as a parameter
def markov_chain(transitions):
    totals = {}
    for pred, succ_counts in transitions.items():
        totals[pred] = sum(succ_counts.values())
    probs = {}
    for pred, succ_counts in transitions.items():
        probs[pred] = {}
        for succ, count in succ_counts.items():
            probs[pred][succ] = count / totals[pred]
    return probs

def generate_Markov(stp, vocabulary,length=10, start=None):
    sentence=[]
    result=''
    # Control of the length of the sentence
    if length>len(stp.keys()):
        raise Exception('The length can not exceed the length tokens long',len(stp.keys()))

    # Control if the state has no successors
    if start not in stp.keys() and start!=None:
        raise Exception('The state given by start is not in the transition probabilities')

    # Control if the starting point is None or other than None state.
    if (start==None):
        special=['.' ,'?','!','.',',',';']
        # select the words from the vocabulary instead of form the probability array
        rand_ini=random.choice(vocabulary)
        while(rand_ini[0] in special):
            rand_ini=random.choice(vocabulary)
        sentence.append(rand_ini)
    else:
        sentence.append(start)

    ls=list(sentence[-1])

    # Generation of words according to the random number and the probability
    for j in range (length):
        # Generate a random number between [0,1] to be evaluated
        rand_choice=random.random()
        acum=0
        # Control if the state has no successors
        if tuple(ls[-len(sentence[0]):]) not in stp.keys():
            break
        for i in stp[tuple(ls[-len(sentence[0]):])].keys():   # For each key of the tuple
            acum += stp[tuple(ls[-len(sentence[0]):])][i]
            if (rand_choice <= acum):  # If the random number belongs to the tuple key
                ls.append(i)   #
                break

    #Concatenation in a single string
    #ls[0]=ls[0].capitalize()
    for word in ls: result+=word+' '
    return result

def random_words(filename, encoding, word_pattern, wlen_limits, n=5):
    '''Returns a list of random words from the file (with length n)
    The 'word' is used loosely here as a word is anything the ``parse_words``
    function will recognize as a word.
    :param str filename: File to learn the words
    :param str encoding: Encoding of the file (most probably 'utf8')
    :param word_pattern: Compiled regex for acceptable words
    :param tuple wlen_limits: Length limits for the acceptable words
    :param int n: Number of words to return
    :returns: a list of the most common (n) words in the file
    '''

    words = parse_words(filename, encoding, word_pattern, wlen_limits)

    # Call Markov's function to calculate the probability matrix with the list of words

    transitionm=transitions(words,m_order)
    prob = markov_chain(transitionm)
    # Generate a random
    result = [random.choice(list(prob.keys()))]
    # Selection of n elements form the probability matrix generated by the ''markov_chain'' function
    for wcount in range(n-1):
        rword=random.choice(list(prob.keys()))
        while(rword in result):
            rword=random.choice(list(prob.keys()))
        result.append(rword)
    # return only the words, not their counts
    return [transitionm, prob,result]

def likelihood(text, stp, order): #stp=state_transitions_probabilities
    # Tokenize the text into sentences.
    sentences = nltk.sent_tokenize(text)
    # Tokenize each sentence to words. Each item in 'words' is a list with
    # tokenized words from that list.
    tokenized_sentences = []
    for s in sentences:
        w = nltk.word_tokenize(s)
        tokenized_sentences.append(w)

    likelihood=0
    for s in range(len(tokenized_sentences[0])-order):
        if tuple(tokenized_sentences[0][s:s+order]) not in stp.keys():
            break
        # Inicialization of the first element and (pseudo)likelihood variable. The probability of the
        # first element is 1/number of tokens
        if likelihood == 0:
            likelihood=(1/len(tokenized_sentences[0]))
        else:
            tostr = str(str(tokenized_sentences[0][s+order:s+order+1]))
            tostr = tostr.replace('[','')
            tostr = tostr.replace(']','')
            tostr = tostr.replace('\'','')
            if tostr not in stp[tuple(tokenized_sentences[0][s:s+order])].keys():
                likelihood == 0
                break
            # Computing the likelihood according the probability of generate the current token given de previous token
            likelihood*=stp[tuple(tokenized_sentences[0][s:s+order])][tostr]
    return likelihood

class ListMemory():
    '''Simple list memory which stores all seen artifacts as is into a list.
    '''
    def __init__(self, capacity):
        '''
        :param int capacity: The maximum number of artifacts in the memory.
        '''
        self._capacity = capacity
        self._artifacts = []

    @property
    def capacity(self):
        '''The maximum number of artifacts in the memory.
        '''
        return self._capacity

    @property
    def artifacts(self):
        '''The artifacts currently in the memory.
        '''
        return self._artifacts

    def memorize(self, artifact):
        '''Memorize an artifact into the memory.

        If the artifact is already in the memory, does nothing. If memory
        is full and a new artifact is memorized, forgets the oldest artifact.

        :param artifact: Artifact to be learned.
        :type artifact: :class:`~creamas.core.artifact.Artifact`
        '''
        if artifact in self._artifacts:
            return

        self._artifacts.insert(0, artifact)
        if len(self._artifacts) > self.capacity:
            self._artifacts = self._artifacts[:self.capacity]

class ToyAgent(CreativeAgent):
    '''A sample agent implementation.
    Agent invents new words be generating them at random and evaluating them
    with respect to its own vocabulary.
    Agent learns its vocabulary from the file given at initialization.
    '''

    def __init__(self, env, filename, encoding='utf8', n=5,
                 wlen_limits=(2,11), chars='abcdefghijklmnopqrstuvwxyz'):
        '''
        :param env:
            subclass of :py:class:`~creamas.core.environment.Environment`
        :param str filename: Filename from which the words should be parsed.
        :param str encoding: Encoding of the file
        :param int n:
            The number of words the agent considers per :func:`invent`
        :param tuple wlen_limits:
            (int, int)-tuple, acceptable word length limits
        :param str chars: acceptable characters in the words
        '''
        super().__init__(env)
        self.n = n
        self.chars = chars
        self.wlen_limits = wlen_limits
        self.word_pattern = re.compile(r'^\w+$')
        # This parta allows that half of the agents in the environment use frequent_words and
        # other half random_words to learn their vocabulary
        randw = random_words(filename, encoding=encoding,
                             word_pattern=self.word_pattern,
                             wlen_limits=self.wlen_limits, n=5)

        self.matrixt = randw[0]
        self.matrixp = randw[1]
        self.vocab = randw[2]
        self.mem = ListMemory(20)
        self.order = m_order
        print('HOLAFIN', self.matrixt)

    def novelty(self, artifact):
        # We will choose that the novelty is maximal if agent's memory is empty.
        if len(self.mem.artifacts) == 0:
            return 1.0, None
        novelty = 1.0
        evaluation_word = artifact.obj
        matching_word = self.mem.artifacts[0].obj
        for memart in self.mem.artifacts:
            word = memart.obj
            lev = levenshtein(evaluation_word, word)
            mlen = max(len(evaluation_word), float(len(word)))
            current_novelty = float(lev) / mlen
            if current_novelty < novelty:
                novelty = current_novelty
                matching_word = word
        return novelty, matching_word

    def evaluate(self, artifact):
        '''Evaluate given artifact with respect to the agent’s vocabulary and the memory.
        Parameters:	artifact – Artifact to be evaluated
        Returns:(evaluation, framing)-tuple, the framing is the combined framing
        of both the value and novelty.'''
        value, value_framing = self.value(artifact)
        novelty, novelty_framing = self.novelty(artifact)
        framing = {'value': value_framing, 'novelty':novelty_framing}
        evaluation = (value + novelty) / 2
        return evaluation, framing

    def value(self, artifact):
        '''Create an evaluation function for the generated pieces of text based on the last week’s
        pseudolikelihood function. (If you have not implemented it, ask from someone who has.)
        That is, the evaluation for the Markov chain is the (pseudo)likelihood of the generated text w.r.t.
        the state transition probabilities. Is this kind of evaluation function desirable?
        '''
        evaluation=0
        # This part permits to perform the evaluation according the matrix of the artifact
        evaluation = likelihood(artifact.obj,self.matrixp,self.order)

        return (evaluation,artifact.obj)

    def learn(self, artifact):
        '''updates the state transition counts with each state transition that is observed
        from the artifact (remember that the actual string was in obj attribute of the
        artifact object).'''
        # tokenize the sentence
        words = artifact.split()
        # calculate transiton matrix of the new information
        prematrix=transitions(words,self.order)
        # add new information to the transition matrix
        for key in prematrix.keys():
            if key in self.matrixt.keys():
                self.matrixt[key].update(prematrix[key])
                #print('The new artifact has strengthen an existing artifact:', artifact)
            else:
                self.matrixt[key]=prematrix[key]
                #print('A new artifact has been learned', artifact)
        # update probability matrix
        self.matrixp = markov_chain(self.matrixt)
        return ()

    def generate(self):
        '''Change your agent to generate text based on Markov chain instead. That is, alter the generate
        function to use Markov chain (either first-order or higher). It is up to you, if you learn the
        state transition probabilities at initialization time for each agent (not advised because of redundancy),
        or do you first learn the state transition probabilities from a single source and then give them
        to each agent at initialization time as a parameter. The pieces of text generated can be of
        fixed (token) length.

        :returns: a word wrapped as :class:`~creamas.core.artifact.Artifact`
        '''
        word_length = random.randint(*self.wlen_limits)
        text = generate_Markov(self.matrixp,self.vocab,word_length)

        return Artifact(self, text, domain=str)

    def invent(self, n=20):
        '''Invent a new word.
        Generates multiple (n) words and selects the one with the highest
        evaluation.
        :param int n: Number of words to consider
        :returns:
            a word wrapped as :class:`~creamas.core.artifact.Artifact` and its
            evaluation.
        '''
        best_artifact = self.generate()
        max_evaluation, framing = self.evaluate(best_artifact) #explicitly renaming the framing returned by evaluate() as framing

        for _ in range(n-1):
            artifact = self.generate()
            evaluation, fr = self.evaluate(artifact)
            # The evaluation is performed according to the likelihood
            if evaluation < max_evaluation:
                best_artifact = artifact
                max_evaluation = evaluation
                framing=fr
        logger.debug("{} invented artifact: {} (eval={}, framing={})"
                     .format(self.name, best_artifact.obj, max_evaluation,framing))
        # Add evaluation and framing to the artifact
        best_artifact.add_eval(self, max_evaluation, fr=framing)
        return best_artifact

    # this definition allows to the artifacts learn from the LibrarianAgent
    @aiomas.expose
    async def service(self,winner):
        self.learn(winner)

    async def act(self):
        '''Agent acts by inventing new words.
        '''
        if len(self.env.artifacts) > 0: # Memorizing the Domain Artifacts
            self.mem.memorize(random.choice(self.env.artifacts))
            #print('arti',self.env.artifacts[-1].obj)
            # Calling to function learn to update the transition matrix
            self.learn(self.env.artifacts[-1].obj)
        artifact = self.invent(self.n) #  We invent a new artifact
        self.mem.memorize(artifact) # Memorize Artifacts
        self.env.add_candidate(artifact) # Add our invented artifact to the voting candidates for this iteration.

class LibrarianAgent(CreativeAgent):
    '''The class defines an agent that executes the communication between the inventors agents. This agent
    "memorizes" the new artifacts, and send the new artifact to the agents which don't belong to the winner
    class (i.e. if the new artifact was produced by an agent that has as source Alice, the Librarian Agent
    will send the article to be memorized and learned by the agents that has as source The War and Peace).'''
    def __init__(self, env):
        super().__init__(env)

    async def act(self):
        artifact = env.artifacts
        if len(artifact) > 0:
            #print ('Artifact',artifact[-1].obj)
            #print ('creator',artifact[-1].creator)
            service_addr=artifact[-1].creator
            service_artifact=artifact[-1].obj

            # Classification of the objects according to their source
            agents=env.get_agents()
            ag_source=[]
            ag_source1=[]
            for name_ag in agents:
                if int(name_ag[-1]) % 2 == 0:
                    ag_source.append(name_ag)
                else:
                    ag_source1.append(name_ag)

            # Artifact is learned by the other source agents
            if service_addr in ag_source:
                #print('Artifact`s Creator:',service_addr, '\nLearned by:',ag_source1)
                for agents1 in ag_source1:
                    service_addr1 = agents1
                    service_agent = await self.env.connect(service_addr1)
                    await service_agent.service(service_artifact)

            if service_addr in ag_source1:
                #print('Artifact`s Creator:',service_addr, '\nLearned by:',ag_source)
                for agents1 in ag_source:
                    if agents1 != 'tcp://localhost:5555/0':
                        service_addr1 = agents1
                        service_agent1 = await self.env.connect(service_addr1)
                        await service_agent1.service(service_artifact)


class ToyEnvironment(Environment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def surprise(self, artifact):
        '''In this case, we have created the function 'surprise' which returns a true value if the number of
        elements of the winner sentence is less than a half of the maximum allowed words.'''
        is_surprise=False
        maxlen=20
        size= len(nltk.word_tokenize(artifact.obj))
        if (size < maxlen/2):
            is_surprise=True
        return (is_surprise)

    def vote(self, age):
        artifacts = self.perform_voting(method='mean')
        if len(artifacts) > 0:
            accepted = artifacts[0][0]
            value = artifacts[0][1]
            # Populating the Domain
            self.add_artifact(accepted) # Add vote winner to domain
            # Evaluation of the surprise of the artifact
            surprising = self.surprise(accepted)
            logger.info("Vote winner by {}: {} (val={}). Surprising:{}"
                        .format(accepted.creator, accepted.obj, value, surprising))
        else:
            logger.info("No vote winner!")
        # To know the vote of each agent
        cands = self.candidates
        votes = {}
        for agent in self.get_agents(address=False):
            votes[agent.name] = agent.vote(cands)
            # print('Votes of agent ',agent.name,':',votes[agent.name])
        self.clear_candidates()


if __name__ == "__main__":
    filename = 'mars.txt'
    filename1 = 'mars.txt'
    env = ToyEnvironment.create(('localhost', 5555))
    m_order= 2
    sever= LibrarianAgent(env)
    for i in range(10):
        if i % 2 == 0:
            m_order= 2
            agent = ToyAgent(env, filename=filename)
        else:
            m_order= 3
            agent = ToyAgent(env, filename=filename1)
    sim = Simulation(env, log_folder='logs', callback=env.vote)
    sim.async_steps(10)
    seq = ''
    for i in range(len(env.artifacts)):
        seq+=(env.artifacts[i].obj)
    print (seq)
    notes1 = NoteSeq(seq)
    midi = Midi(1, tempo=90)
    midi.seq_notes(notes1, track=0)
    midi.write("demo.mid")
    sim.end()