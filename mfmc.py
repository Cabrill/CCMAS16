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
from music_agent import MusicAgent

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
                    MusicAgent(env, textmc, textst, helper, list_memory, server.addr)
    
    while len(env.get_agents()) < num_agents:
        print("Creating leftover agents for " + text_file)
        MusicAgent(env, textmc, textst, helper, list_memory, server.addr)

    #Start the simulation
    sim = Simulation(env, log_folder='logs', callback=env.vote)
    logger.info("Initialization complete!  Starting simulation...")
    sim.async_steps(voting_rounds)
    sim.end()
    csvFile.close()