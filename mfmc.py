'''
MFMC(Music From Multiple Content) project to create music(lyrics+tracks) using multiple agents in a `creamas <https://github.com/assamite/creamas/>`_ environment and using the `pyknon <http://kroger.github.io/pyknon/>`_ library
to create MIDI files.
'''
import logging,random,numpy,re,string,aiomas,nltk,os,subprocess, platform
import argparse
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
    '''Agent which allows MusicAgents to request the address of another
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
    '''
    An environment for our MusicAgents to interact.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def vote(self, age):
        '''
        Perform the act of voting on this rounds artifacts to select the best one available
        from all participating agents.
        '''
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
        '''
        Saves the track list in the artifact to a MIDI file
        
        :param artifact: :class:`~creamas.core.Artifact` containing the track list to be written
        '''
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
        '''
        Plays the MIDI file at the specified path.
        
        :param str filepath: The filename of the MIDI file to play
        '''
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
    '''
    Reads the contents of the Inspiring Set folder to find TXT files.
    
    :returns:
        a list of TXT file names of files contained in the path
    '''
    text_files = [f  for f in listdir(mypath) if isfile(join(mypath, f)) and str(f).endswith(".txt")]
    return text_files
    
if __name__ == "__main__":  
    #Read in any supplied user arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agents", help="The number of concurrent agents to simulate.  DEFAULT: 5", type=int)
    parser.add_argument("-p", "--path", help="The location of the inspiring set of TXT files. DEFAULT: InspiringSet", type=str)
    parser.add_argument("-o", "--order", help="The order of Markov Chain to use in lyric generation.  DEFAULT: 2", type=str)
    parser.add_argument("-r", "--rounds", help="The number of voting rounds to simulate.  DEFAULT: 1000", type=int)
    parser.add_argument("-m", "--memory", help="The number of seen artifacts an agent can remember.  DEFAULT: 20", type=int)
    args = parser.parse_args()
    
    #Set Inspiring Set path (location of TXT files)
    if args.path:
        path = args.path
    else:
        path = "InspiringSet/"
    
    #Set the number of agents
    if args.agents:
        num_agents = args.agents
    else:
        num_agents = 5
        
    #Set the Markov Chain order
    if args.order:
        order = args.order
    else:
        order = 2
    
    #Set the memory list length
    if args.memory:
        list_memory = memory
    else:
        list_memory = 20
        
    #Set the number of rounds
    if args.rounds:
        voting_rounds = args.rounds
    else:
        voting_rounds = 1000
        
    logger.info("Initializing environment and agent data...")
    
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
            textmc, textst = markov_chain(text_read, order)
            
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