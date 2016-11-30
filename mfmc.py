'''
.. py:module:: mfmc
    :platform: Unix

Toy example of using `creamas <https://github.com/assamite/creamas/>`_ to build
a multi-agent system.
'''
import logging,random,numpy,re,string,aiomas,nltk,os,subprocess, platform
from os import listdir
from os.path import isfile, join
from collections import Counter
from mc import markov_chain, generate, sanitize, determineOrder, likelihood, format_for_printing
from list_mem import ListMemory
from creamas.core import CreativeAgent, Environment, Simulation, Artifact
from serializers import get_artifact_ser
from music_util import MusicHelper
from pyknon.genmidi import Midi
from pyknon.music import NoteSeq

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
    
class ToyAgent(CreativeAgent):
    '''A sample agent implementation.

    Agent invents new words be generating them at random and evaluating them
    with respect to its own vocabulary. 

    Agent learns its vocabulary from the file given at initialization.
    '''

    def __init__(self, env, mcprobs, mcstates, helper, mem_len, service_addr, encoding='utf8', n=20,
                 wlen_limits=(30,100), chars='abcdefghijklmnopqrstuvwxyz'):
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
        self.MarkovChainProbs = mcprobs
        self.MarkovChainStates = mcstates
        self.mcpOrder = determineOrder(mcprobs)
        self.mem = ListMemory(mem_len)
        self.service_addr = service_addr
        self.music_helper = helper
        
    def learn(self, artifact):
        '''Improve the agents Markov Chain by incorporating the state transitions
        in the supplied artifact.
        
        :param artifact: :class:`~creamas.core.Artifact` to be evaluated
        '''
        raw_text = artifact.obj[0]
        
        # Tokenize the text into words
        tokenized_words = nltk.word_tokenize(raw_text)
        
        # Use the same order as existing learned Markov Chain
        order = self.mcpOrder
        
        # Now we are ready to create the state transitions
        for data in enumerate(tokenized_words):
            for i in range(len(data[1])-order):
                pred = data[1][i]
                for j in range(1, order):
                    pred += " " + data[1][i+j]
                succ = data[1][i+order]
                if pred not in self.MarkovChainStates:
                    # Predecessor key is not yet in the outer dictionary, so we create
                    # a new dictionary for it.
                    self.MarkovChainStates[pred] = {}
                if succ not in self.MarkovChainStates[pred]:
                    # Successor key is not yet in the inner dictionary, so we start
                    # counting from one.
                    self.MarkovChainStates[pred][succ] = 1.0
                else:
                    # Otherwise we just add one to the existing value.
                    self.MarkovChainStates[pred][succ] += 1.0

    def compute_probabilities(self):
        # Compute total number of successors for each state
        totals = {}
        for pred, succ_counts in self.MarkovChainStates.items():
            totals[pred] = sum(succ_counts.values())
        
        # Recompute the probability for each successor given the predecessor.
        for pred, succ_counts in self.MarkovChainStates.items():
            if not pred in self.MarkovChainProbs:
                self.MarkovChainProbs[pred] = {}
            for succ, count in succ_counts.items():
                self.MarkovChainProbs[pred][succ] = count / totals[pred]

    def grammar_check(self, artifact):
        '''This is meant to be a simple way to exclude artifacts that
        end in a word known to be invalid ending for a sentence.
        '''
        score = 1
        words = nltk.word_tokenize(artifact.obj[0])
        if ''.join(words[-1:]) in ("and", "the", "of", "if", "their", "a", "as", "but" ):
            score = 0
        return score
        
    def value(self, artifact):
        '''Evaluate given artifact with respect to the probability of it 
        occurring using the state transition probabilities from the 
        markov chain that the agent knows.
        
        Actual evaluation formula for a string is:
        
        probability = 1 - likelihood where likelihood is the sum product of all tokens probability
            according to this agents Markov Chain.
            
        length = Token length / maximum token length.  This favors longer artifacts.
        
        grammar = Simple grammar check, that invalidates an artifact if it fails.
       
        :param artifact: :class:`~creamas.core.Artifact` to be evaluated
        :returns:
            (evaluation, word)-tuple, containing both the evaluation and the
            evaluation method
        '''
        probability = 1 - likelihood(artifact.obj[0], self.MarkovChainProbs)
        length = len(nltk.word_tokenize(artifact.obj[0])) / self.wlen_limits[1]
        grammar = self.grammar_check(artifact)
        evaluation = grammar * probability * length
        return evaluation, "MarkovChainEvaluation"
    
    def novelty(self, artifact):
        '''Assign a value for the novelty of an artifact.  This value is based one
        the similarity of the artifact to previously recorded artifacts and
        therefore represents how unique the artifact is to this agent.   
        
        :param artifact: :class:`~creamas.core.Artifact` to be evaluated
        :returns:
            (noveltyscore, method)-tuple, containing both the novelty score and the
            method used to reach this score
        '''
        # We will choose that the novelty is maximal if agent's memory is empty.
        if len(self.mem.artifacts) == 0:
            return 1.0, None

        novelty = 1.0
        evaluation_word = artifact.obj[0]
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
        '''Evaluate given artifact by checking its inherent value and novelty.
        
        :param artifact: :class:`~creamas.core.Artifact` to be evaluated
        :returns:
            (evaluation, word)-tuple, containing both the evaluation and the
            evaluation method
        '''
        value, value_framing = self.value(artifact)
        novelty, novelty_framing = self.novelty(artifact)
        framing = {'value': value_framing, 'novelty':novelty_framing}
        evaluation = (value + novelty) / 2
        
        return evaluation, framing

    def generate(self):
        '''Generate new text.

        Text is generated by state transition probability of a markov chain. Text length is
        in ``wlen_limits``.

        :returns: a string of text wrapped as :class:`~creamas.core.artifact.Artifact`
        '''
        text_length = random.randint(*self.wlen_limits)
        text = generate(self.MarkovChainProbs, text_length)
        
        derived_notes = self.music_helper.convert_phrase_to_notes(text)
        music_key = self.music_helper.determine_dominant_key(derived_notes)
        conformed_notes = self.music_helper.conform_notes_to_key(derived_notes, music_key)
        
        tuple_artifact = (text, conformed_notes)
        
        return Artifact(self, tuple_artifact, domain=tuple)

    def invent(self, n=20):
        '''Invent a new word.

        Generates multiple (n) words and selects the one with the highest
        evaluation.

        :param int n: Number of words to consider
        :returns:
            a word wrapped as :class:`~creamas.core.artifact.Artifact` and its
            evaluation.
        '''
        # Before we generate a new artifact, we should incorporate our learning
        # by recomputing the Markov Chain probabilities
        self.compute_probabilities()
        
        best_artifact = self.generate()
        max_evaluation, framing = self.evaluate(best_artifact)
        for _ in range(n-1):
            artifact = self.generate()
            evaluation, fr = self.evaluate(artifact)
            if evaluation > max_evaluation:
                best_artifact = artifact
                max_evaluation = evaluation
                framing = fr
        '''logger.debug("{} invented word: {} (eval={}, framing={})"
                     .format(self.name, best_artifact.obj, max_evaluation,
                             framing))
        '''
        # Add evaluation and framing to the artifact
        best_artifact.add_eval(self, max_evaluation, fr=framing)
        return best_artifact


    async def act(self):
        '''Agent acts by inventing new artifacts.
        '''
        #If there are artifacts available in the domain, learn from one
        if len(self.env.artifacts) > 0:
            random_artifact = random.choice(self.env.artifacts)
            self.mem.memorize(random_artifact)
            self.learn(random_artifact)
                        
        #Now invent a new one
        artifact = self.invent(self.n)
        self.mem.memorize(artifact)
        self.learn(artifact)
        
        #Ask the service for the address of a random agent to contact
        service_agent = await self.env.connect(self.service_addr)
        agent_address = await service_agent.request_random_agent()
        
        #If the random agent isn't itself, then contact the agent to ask 
        #for an opinion and to get one of their artifacts
        if not agent_address == self.addr:
            random_agent = await self.env.connect(agent_address)
            their_opinion, requested_artifact = await random_agent.request_artifact_exchange(artifact)
        
            #Make an opinion on their artifact
            my_opinion = self.value(requested_artifact)[0]
        
            #If this agent liked the returned artifact, then learn from it
            if my_opinion > 0.85:
                self.mem.memorize(requested_artifact)
                self.learn(requested_artifact)
                
            #Also, if the other agent didn't like this one's artifact, try once more
            if their_opinion < 0.85:
                artifact = self.invent(self.n)
        
        

        #Now give the environment the final artifact for voting
        
        self.env.add_candidate(artifact)
        
    @aiomas.expose
    async def request_artifact_exchange(self, senders_artifact):
        '''Allows an agent to send an artifact to another agent in exchange
        for their opinion on the artifact, and an artifact of their own.
        '''
        my_opinion = self.value(senders_artifact)[0]
        
        #Learn from this artifact if it's liked and return the favor
        if my_opinion > 0.85:
            self.mem.memorize(senders_artifact)
            self.learn(senders_artifact)
            my_artifact = self.generate()
        else:
            #If their artifact was not good, give them something random from memory
            #if one is available, and if it's not then create one.
            if len(self.mem.artifacts) > 0:
                my_artifact = random.choice(self.mem.artifacts)
            else:
                my_artifact = self.generate()
        
        return my_opinion, my_artifact

class ServiceAgent(CreativeAgent):
    '''Agent which defines a service for other agents to use.
    '''

    def __init__(self, env):
        super().__init__(env)

    @aiomas.expose    
    async def request_random_agent(self):
        '''Allows an agent to get access to a random agent to request an artifact exchange
        Note:  At this time, an agent can be given itself. 
        '''
        all_agents = [agent for agent in self.env.get_agents(address=False) if not isinstance(agent, ServiceAgent)]
        random_agent = random.choice(all_agents)
            
        return random_agent.addr
        
    async def act(self):
        return
        
class ToyEnvironment(Environment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def vote(self, age):
        artifacts = self.perform_voting(method='mean')
        if len(artifacts) > 0:
            accepted = artifacts[0][0]
            accepted_value = artifacts[0][1]
            self.add_artifact(accepted) # Add vote winner to domain
            formatted_result = format_for_printing(accepted.obj[0]) + "\n" + str(accepted.obj[1])
            logger.info("Vote winner by {}: {} (val={})"
                        .format(accepted.creator, formatted_result, accepted_value))
        else:
            logger.info("No vote winner!")

        self.clear_candidates()
        file_name = self.save_midi(accepted)
        self.play_midi(file_name)
        input("Playing artifact's music.  Press a key for next voting round...")
        
    def save_midi(self, artifact):
        notes = artifact.obj[1]
        midi = Midi(1, tempo=90)
        midi.seq_notes(notes, track=0)
        file_name = "result" + str(self.age) + ".mid"
        midi.write(file_name)
        return file_name
        
    def play_midi(self, filepath):
        if platform.system().lower().startswith('darwin'):
            subprocess.call(('open', filepath))
        elif os.name == 'nt':
            os.startfile(filepath)
        elif os.name == 'posix':
            subprocess.call(('xdg-open', filepath))
        
def getTextFromFile(filename):
    '''
    Opens the specified file at filename and returns the raw text
    
    :param str filename: The name of the file to read
    :returns:
        a string containing the raw text
    '''
    with open(filename, 'r', encoding='utf8') as f:
        raw_text = f.read()

    # And replace more than one subsequent whitespace chars with one space
    raw_text = re.sub(r'\s+', ' ', raw_text)
    return raw_text;
    
def read_text(self, mypath="InspiringSet/"):
    text_files = [f  for f in listdir(mypath) if isfile(join(mypath, f))]
    return text_files
    
if __name__ == "__main__":
    logger.info("Initializing environment and agent data...")
    
    #Set Inspiring Set path (location of TXT files)
    path = "InspiringSet/"
    
    #Set the memory length and voting rounds
    list_memory = 25
    voting_rounds = 1000
    num_agents = 5
    
    #Create the simulation environment
    env = ToyEnvironment.create(('localhost', 5555), codec=aiomas.MsgPack, extra_serializers=[get_artifact_ser])
    
    #Create a service agent
    server = ServiceAgent(env)
    
    #Create a music helper for determining scales
    helper = MusicHelper()
    
    #Check for available text
    text_file_list = read_text(path)
    num_text = len(text_file_list)
    
    if num_text < 1:
        raise ValueError("No text was found in the '" + path + "' location.")
    
    if num_agents < num_text:
        agents_per_text = 1
    else:
        agents_per_text = int(num_agents / num_text)

    for text_file in text_file_list:
        if len(env.get_agents()) < num_agents:
            logger.info("Reading '"+text_file+"'...")
            fqpn = path + text_file
            #Read the text data
            text_read = getTextFromFile(fqpn)
            #Learn the Markov Chains from the text
            textmc, textst = markov_chain(text_read)
            
            for i in range(0, agents_per_text):
                if len(env.get_agents()) < num_agents:
                    print("Creating an agent for " + text_file)
                    ToyAgent(env, textmc, textst, helper, list_memory, server.addr)
    
    while len(env.get_agents()) < num_agents:
        print("Creating leftover agents for " + text_file)
        ToyAgent(env, textmc, textst, helper, list_memory, server.addr)

    #Start the simulation
    sim = Simulation(env, log_folder='logs', callback=env.vote)
    logger.info("Initialization complete!  Starting simulation...")
    sim.async_steps(voting_rounds)
    sim.end()
    csvFile.close()