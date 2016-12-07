'''
.. py:module:: mfmc
    :platform: Unix

MFMC(Music From Multiple Content) project to create music(lyrics+tracks) using multiple agents in a `creamas <https://github.com/assamite/creamas/>`_ environment and using the `pyknon <http://kroger.github.io/pyknon/>`_ library
to create MIDI files.
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
        
class MusicEnvironment(Environment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def vote(self, age):
        artifacts = self.perform_voting(method='mean')
        if len(artifacts) > 0:
            accepted = artifacts[0][0]
            accepted_value = artifacts[0][1]
            self.add_artifact(accepted) # Add vote winner to domain
            lyrics = accepted.obj[0]
            theme = accepted.obj[1]
            logger.info("Winning song created by: {} \nLyrics:{} \nTheme based on: {} \n(val={})"
                        .format(accepted.creator, lyrics, theme, accepted_value))

            logger.info("Tempo: " + str(accepted.obj[2][0]))
            logger.info("Tracks: "+ str(len(accepted.obj[3])))
            instr1 = MusicHelper.determine_instrument(accepted.obj[2][1][0])
            instr2 = MusicHelper.determine_instrument(accepted.obj[2][1][1])
            instr3 = MusicHelper.determine_instrument(accepted.obj[2][1][2])
            
            logger.info("Instrument 1: " + str(accepted.obj[2][1][0]+1) + " " + instr1[0] + "-" + instr1[1])
            if len(accepted.obj[3]) > 1:
                logger.info("Instrument 2: " + str(accepted.obj[2][1][1]+1) + " " + instr2[0] + "-" + instr2[1])
            if len(accepted.obj[3]) > 2:
                logger.info("Instrument 3: " + str(accepted.obj[2][1][2]+1) + " " + instr3[0] + "-" + instr3[1])
        else:
            logger.info("No vote winner!")

        self.clear_candidates()
        file_name = self.save_midi(accepted)
        self.play_midi(file_name)
        input("Playing artifact's music.  Press a key for next voting round...")
        
    def save_midi(self, artifact):
        track_list = artifact.obj[3]
        music_theme = artifact.obj[2]
        word_theme = artifact.obj[1]
        
        midi = Midi(number_tracks = len(track_list), tempo=music_theme[0], instrument=music_theme[1])
        for i in range(0, len(track_list)):
            midi.seq_notes(track_list[i], track=i)
        
        file_name = word_theme + str(self.age) + ".mid"
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
    text_files = [f  for f in listdir(mypath) if isfile(join(mypath, f)) and str(f).endswith(".txt")]
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
    env = MusicEnvironment.create(('localhost', 5555), codec=aiomas.MsgPack, extra_serializers=[get_artifact_ser])
    
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