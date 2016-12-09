'''
A MusicAgent creates lyrics and music using its uniquely personal invention methods which are created
usings lyrics it has generated from its inspiring set.  Music is generated based on lyrics, using the
invention method it matches to it based on the probability of the lyrics occurring using its Markov Chain
probabilities.
'''

import aiomas, re, random, nltk, operator
from creamas.core import CreativeAgent, Environment, Simulation, Artifact
from mc import markov_chain, generate, sanitize, determineOrder, likelihood, format_for_printing
from list_mem import ListMemory
from pyknon.music import NoteSeq, Rest
from collections import defaultdict

def levenshtein(s, t):
    '''Compute the edit distance between two strings.

    From Wikipedia article; Iterative with two matrix rows.
    :param string: The origination string to be evaluated
    :param string: The destination string to be compared against the origination
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
    
class invention_method:
    '''
    A list of functions used to generate parameters in the creation of, 
    and minimal statistics about the overall invention method
    '''
    def __init__(self, method_list):
        '''        
        :param list method_list: A list of methods accepting a string, used for invention
        '''
        self.times_utilized = 0
        self.average_rating = 1
        self.method_list = method_list

class MusicAgent(CreativeAgent):
    '''An agent that creates music in the form of lyrics and instrument tracks.

    Agent invents new lyrics based on their Markov Chain probabilities computed
    from inspecting their assigned TXT from the inspiration set at initialization.
    Which is then modified when learning from new artifacts created by other agents.
    '''

    def __init__(self, env, mcprobs, mcstates, helper, mem_len, service_addr, n=20,
                 wlen_limits=(3,10), method_limits = 5):
        '''
        :param env:
            subclass of :py:class:`~creamas.core.environment.Environment`
   
        :param dictionary mcprobs:  
            The Markov Chain probabilities learned from the text
        
        :param dictionary mcstates: 
            The Markov Chain state transitions learned from the text
        
        :param MusicHelper helper: 
            A class with generic methods used to generate music from lyrics
        
        :param int mem_len: 
            The maximum number of artifacts to store in agent's memory
        
        :param str service_addr:  
            The address of the service_agent used to contact other agents
        
        :param int n:
            The number of words the agent considers per :func:`invent`

        :param tuple wlen_limits:
            (int, int)-tuple, acceptable word length limits
            
        :param int method_limits:
            The maximum number of invention methods this agent may keep concurrently
        '''
        super().__init__(env)
        self.n = n
        self.wlen_limits = wlen_limits
        self.MarkovChainProbs = mcprobs
        self.MarkovChainStates = mcstates
        self.mcpOrder = determineOrder(mcprobs)
        self.mem = ListMemory(mem_len)
        self.service_addr = service_addr
        self.music_helper = helper
        self.method_limit = method_limits
        self.invention_methods = list()
        
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
        '''Computes the probabilities of Markov Chain items representing state transitions from
        word string of token length order to the next word.
        '''
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
        
        :param artifact: :class:`~creamas.core.Artifact` containing the lyrics to be evaluated
        :returns: A score of the artifact's employed grammar, from 0 to 1.
        '''
        score = 1
        words = nltk.word_tokenize(artifact.obj[0])
        if ''.join(words[-1:]) in ("and", "the", "of", "if", "their", "a", "as", "but", "or" ):
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
        
        music = Evaluation of use of patterns and instrument choices
       
        :param artifact: :class:`~creamas.core.Artifact` to be evaluated
        :returns:
            (evaluation, word)-tuple, containing both the evaluation and the
            evaluation method
        '''
        probability = 1 - likelihood(artifact.obj[0], self.MarkovChainProbs)
        length = len(nltk.word_tokenize(artifact.obj[0])) / (self.wlen_limits[1] * 4)
        grammar = self.grammar_check(artifact)
        music = self.eval_music(artifact)
        evaluation = grammar * probability * length * music
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

    def eval_music(self,artifact):
        '''Attempts to evaluate the quality of music by looking at instrument selection
        and pattern repetition.
        
        :param artifact: :class:`~creamas.core.Artifact` containing the music to be evaluated
        :returns: double value from 0 to 1 representing the quality of music
        '''
        #evaluate instruments

        #according to midi standard sounds list(see http://soundprogramming.net/file-formats/general-midi-instrument-list/) we decided:
        # Lead:
        # 1: 8, 25: 31, 41: 43, 57, 65: 72, 73: 80, 105: 112
        #
        # Rhythm:
        # 17: 24, 25: 31, 33: 40, 41: 48, 49: 56, 57: 64,
        #
        # Percussion:
        # 9: 16, 81: 88, 89: 96, 97: 104, 113: 120
        #
        # bad
        # 32, 121: 128
        inst_eval = 0
        lead = list(range(8,25))+ list(range(25,31)) + list(range(41,43)) + list(range(65,80)) + list(range(105,112)) #appropriate lead sounds
        lead.append(57)
        rhythm = list(range(17,24)) + list(range(25,31)) +list(range(33,64)) #appropriate rhythm sounds
        percussion = list(range(9,16)) + list(range(81,104)) + list(range(113,120)) #appropriate percussion sounds
        bad = list(range(121,128)) #inappropriate sounds for a song
        bad.append(32)
        #first we return the midi id of the instruments
        ins1 = artifact.obj[2][1][0] + 1
        ins2 = artifact.obj[2][1][1] + 1
        ins3 = artifact.obj[2][1][2] + 1
        if ins1 in bad: #bad score for bad sounds
            inst_eval -= 3
        elif ins1 in lead: #best score for lead sounds
            inst_eval += 2
        elif ins1 in rhythm: #good score for rhythm sounds
            inst_eval += 1
        else: #bad score for percussions
            inst_eval -= 2

        if ins2 in bad: #bad score for bad sounds
            inst_eval -= 3
        elif ins2 in lead: #good score for lead sounds
            inst_eval += 1
        elif ins2 in rhythm: #best score for rhythm sounds
            inst_eval += 2
        else: #bad score for percussions
            inst_eval -= 2

        if ins3 in bad: #worst score for bad sounds
            inst_eval -= 3
        elif ins3 in percussion: #best score for percussions
            inst_eval += 2
        else: #bad score for rhythm or lead sounds
            inst_eval -= 2


        #evaluate melody
        def sanitize_track(artifact): #remove all not-note elements of the track_list in order to evaluate only the notes sequence
            tr_list = artifact.obj[3]
            line = str(tr_list)
            line = re.sub('[<>:R.#, ]', '', line)
            line = ''.join([i for i in line if not i.isdigit()])
            line = line.replace("Seq", "")
            line = line.replace("[", "")
            line = line.replace("]", "")
            return line

        minPhrase = 12 #min notes that a phrase should have
        notes = sanitize_track(artifact) #only the notes of track_list
        occur = {} #dictionary to cotain all possible phrases and their number of occurrences
        for i in range(len(notes) - minPhrase + 1):
            select = notes[i:i + minPhrase] #selected phrase to look for
            if select not in occur: #if the phrase is not in the dictionary, add it
                occur[select] = 1 #inital value of occurrence

            rest = notes[0:i] + notes[i + minPhrase:len(notes)] #rest of the notes without the selected phrase
            if occur[select] <= 1: #if we haven't looked for the phrase, look for it, else continue to the next one
                for j in range(len(rest) - minPhrase + 1):
                    if rest[j:j + minPhrase] == select: #if selected phrase is found, increase its counter by one
                        occur[select] += 1
        sorted_occ = sorted(occur.items(), key=operator.itemgetter(1), reverse=True) #sort the occurrence dictionary in descending order
        #sorted_occ[i] returns the most repeated phrase and its number of repetitions. e.g. ('aba', 3).
        # If the number of repetitions is <= 1 then there were no repeated phrases. Because the occurences dictionary is sorted
        #as a list, we can return more elements if needed
        
        #Convert these phrase counts to a 0-1 evaluation
        eval_score = 0
        ideal_count = 5
        for i in range(min(4, max(0,len(sorted_occ)-1)), 0,  -1):
            if len(sorted_occ[i]) > 1:
                phrase_count = sorted_occ[i][1]
                phrase_score = max(0,(ideal_count - abs(phrase_count - ideal_count)) / ideal_count) #Measures proximity to ideal
                phrase_score = phrase_score * (0.0715 * (i+1))#This 0.0715*i gives depreceating returns for more patterns
                eval_score = eval_score + phrase_score
        
        eval_score = min(1, eval_score) #Correct rounding error that would result in 1.01

        #add instrument eval to eval_score
        eval_score += inst_eval
        
        return eval_score

    def generate(self):
        '''Generate new text.

        Text is generated by state transition probability of a markov chain. Text length is
        in ``wlen_limits``.

        :returns: a string of text wrapped as :class:`~creamas.core.artifact.Artifact`
        '''
        lyrics = ""
        for i in range(0,3):
            text_length = random.randint(*self.wlen_limits)
            lyrics = lyrics + " " + format_for_printing(generate(self.MarkovChainProbs, text_length))
        lyrics = lyrics.strip()
        
        #Choose an invention method for the song and record its index for later evaluation
        invention_method = self.choose_invention_method(lyrics)
        method_used_idx = self.invention_methods.index(invention_method)
        
        #Create a word theme, music theme, and track_list
        word_theme, music_theme, track_list = self.create_music(lyrics, invention_method)
        
        #Create an artifact containing the creations
        tuple_artifact = (lyrics, word_theme, music_theme, track_list, method_used_idx)
        
        return Artifact(self, tuple_artifact, domain=tuple)
        
    def create_music(self, lyrics, invention_method):
        '''Generate music from text.

        Music is generated by deriving notes from text, harmonizing those notes, and splitting them
        into separate tracks, choosing a theme based on the text and then matching the tracks up.

        :returns: word_theme=a string representing inspiration, music_theme=tempo and instrument list, track_list=list of tracks composed of notes
        '''
        #Read in characters in the lyrics, and convert to musical notes
        derived_notes = self.music_helper.convert_phrase_to_notes(lyrics)
        #Determine which musical key is dominant in the derived notes
        music_key = self.music_helper.determine_dominant_key(derived_notes)
        #Force all notes into the dominant key so we don't have dischord
        notes = self.music_helper.conform_notes_to_key(derived_notes, music_key)
        
        #Tokenize the word list
        lyric_words = nltk.word_tokenize(lyrics)
        
        #Find a word that will provide a theme for the song
        music_theme = None
        for i in range(0, len(lyric_words)):
            if music_theme != None:
                break
            word_theme = lyric_words[i]
            if len(word_theme) > 2:
                music_theme = self.music_helper.determine_theme(word_theme)
        #No matching words were found, choose one at random
        if music_theme == None:
            word_theme = "random"
            music_theme = self.music_helper.determine_theme(word_theme)

        track_list = []
        lead_track = NoteSeq()
        other_notes = NoteSeq()
        
        lead_note_duration = (invention_method.method_list[0](lyrics) % 6) * 0.05
        lead_rest_duration = (invention_method.method_list[1](lyrics) % 8) * 0.25
        
        #Separate notes into lead track/others, assign word-based duration
        for i in range(0,len(notes)):
            #Associate each note for the lead track with a word in the lyrics,
            #until we're out of words, then put the rest in "other"
            if i < len(lyric_words):
                word_ptr = i
                word_for_note = lyric_words[word_ptr]
                
                #Exclude punctuation in lead track
                if word_for_note not in (',', '.', ';', '!', '?', '"', ':', '/', '\\'):
                    #Set notes duration based on word length
                    notes[i].dur = len(word_for_note) * lead_note_duration
                    lead_track.append(notes[i])
            else:
                other_notes.append(notes[i])
                
        #Insert rests for lead track at punctuation marks
        rest_count = 0
        for i in range(0, len(lead_track)):
            if lyric_words[i] in (',', '.', ';', '!', '?', '"', ':', '/', '\\'):
                lead_track.insert(i+rest_count, Rest(lead_rest_duration))
                rest_count = rest_count + 1
        
        #See how long the lead track is
        lead_track_duration = sum([noteOrRest.dur for noteOrRest in lead_track])      
        #Then add it to our track list
        track_list.append(lead_track)
        
        #If there aren't enough notes, add some from the lead track
        if len(other_notes) < 8:
            lead_length = len(lead_track)
            for i in range(lead_length-1,max(0,lead_length-8), -1):
                if not isinstance(lead_track[i], Rest):
                    other_notes.append(lead_track[i])
    
        #Attempt to detect patterns in the lyrics in combination with the
        #other notes, for the purpose of creating more tracks using the agent's 
        #preferred invention method
        if len(other_notes) > 0:
            pattern_tracks = self.music_helper.derive_tracks_from_lyrics(lyrics, other_notes, lead_track_duration, invention_method.method_list)
        for i in range(0, len(pattern_tracks)):
            track_list.append(pattern_tracks[i])
                
        #Find out which track came out the longest in duration
        longest_duration = lead_track_duration
        for i in range(1, len(track_list)):
            this_track_duration = sum([noteOrRest.dur for noteOrRest in track_list[i]])
            if this_track_duration > longest_duration:
                longest_duration = this_track_duration

        #Make the tracks equal in duration, so there isn't long silence                
        for i in range(0, len(track_list)):
            #Calculate this track duration
            this_track_duration = sum([noteOrRest.dur for noteOrRest in track_list[i]])
            #Add some rests before/during to make it centered
            if this_track_duration < longest_duration:
                insert_rest = (longest_duration - this_track_duration) / 2
                track_list[i].insert(0,Rest(insert_rest))
                track_list[i].append(Rest(insert_rest))
            
            #Add a 2 second pause to the end of the longest track so it ends gracefully
            if this_track_duration == longest_duration:
                track_list[i].insert(0,Rest(2))
            
        return word_theme, music_theme, track_list
            
    def reconsider_invention_methods(self, lyrics):
        '''
        Allows an agent to create new invention methods, or replace a poorly performing method.
        :param string lyrics: Lyrics to be used as inspiration for generation of a new method
        '''
        #First check if we've filled our maximum number of methods
        if len(self.invention_methods) < self.method_limit:
            self.invention_methods.append(self.create_an_invention_method(lyrics))
        else:
            #Identify the worst performing invention method that has been used at least five times
            worst_method = None
            worst_rating = 1
            for invention_method in self.invention_methods:
                if invention_method.times_utilized > 5 and invention_method.average_rating < worst_rating:
                    worst_method = invention_method
                    break
            
            #Replace the worst method, if one found
            if worst_method:
                #Remove this poor performer from invention methods
                self.invention_methods.remove(worst_method)
                #Replace it with something new
                self.invention_methods.append(self.create_an_invention_method(lyrics))

    def choose_invention_method(self, lyrics):
        ''' 
        Agent chooses the best available invention method they know, with randomness provided by
        using the probability of the first two words in the lyrics as a seed.
        :param lyrics: string to be used for generation of a new method, if necessary
        '''
        #First check we've create an invention method, and create a new one to use if none exist
        if len(self.invention_methods) < 1:
            self.invention_methods.append(self.create_an_invention_method(lyrics))
            return self.invention_methods[0]
        else:
            #Otherwise, choose the best one from those available, using the lyrics as a seed for decision
            lyrics_tokens = nltk.word_tokenize(lyrics)
            lyric_sample = ' '.join(lyrics_tokens[0:1])
            lyric_seed = 1 - likelihood(lyric_sample, self.MarkovChainProbs)
            best_method = None
            self.invention_methods.sort(key=lambda method_list: method_list.average_rating, reverse=True)
            for invention_method in self.invention_methods:
                if invention_method.average_rating >= lyric_seed:
                    best_method = invention_method
                    break
            #If no method chosen based on the lyrics, choose the highest ranked one 
            if not best_method:
                best_method = self.invention_methods[0]
                if len(self.invention_methods) < self.method_limit:
                    best_method = self.create_an_invention_method(lyrics)
                    self.invention_methods.append(best_method)
                else:
                    best_method = self.invention_methods[0]

            return best_method        
    
    def create_an_invention_method(self, lyrics):
        '''
        Agent creates a new invention method, using the current lyrics as a seed of inspiration
        :param lyrics: string to be used for generation of a new method, if necessary
        :returns: A list of invention_methods.
        '''
        method_list = []
        lyrics_tokens = nltk.word_tokenize(lyrics)
        token_len = len(lyrics_tokens)
        #Create five methods used to derive values used in music generation
        for method_ref in range(0, 5):
            #Use the probability of a sample of the lyrics as a seed for creation
            lyric_index = min(token_len-2, method_ref % (token_len-1))
            lyric_sample = ' '.join(lyrics_tokens[lyric_index:lyric_index+1])
            lyric_seed = 1 - likelihood(lyric_sample, self.MarkovChainProbs)
            
            if lyric_seed >= 0.9:
                new_method = lambda text: max(1,len(text) % (method_ref+5))
            elif lyric_seed >= 0.8:
                new_method = lambda text: max(1,ord(text[0]) % (43+method_ref))
            elif lyric_seed >= 0.7:
                new_method = lambda text: max(1,ord(text[-1]) % (43+method_ref))
            elif lyric_seed >= 0.6:
                new_method = lambda text: max(1,ord(text[0]) + ord(text[-1]) % (method_ref+43))
            elif lyric_seed >= 0.5:
                new_method = lambda text: max(1,len(text) % max(3,(ord(text[0])%43)))
            elif lyric_seed >= 0.4:
                new_method = lambda text: max(1,sum([ord(c) for c in text]) % (43+method_ref))
            elif lyric_seed >= 0.3:
                new_method = lambda text: max(1,abs(ord(text[0])-ord(text[1]) % (method_ref+5)))
            elif lyric_seed >= 0.2:
                new_method = lambda text: max(1,abs(ord(text[0])-ord(text[-1]) % (method_ref+5)))
            elif lyric_seed >= 0.1:
                new_method = lambda text: max(1,abs(len(text)% abs(ord(text[0])-43) - method_ref))
            else:
                new_method = lambda text: max(1,abs(len(text)% abs(ord(text[-1])-43) - method_ref))
        
            method_list.append(new_method)
        
        return invention_method(method_list)
        
    def invent(self, n=10):
        '''Invents a new sentence

        Generates multiple (n) words and selects the sentence with the highest rating.

        :param int n: Number of sentences to consider
        :returns:
            a sentence wrapped as :class:`~creamas.core.artifact.Artifact` and its
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

        # Add evaluation and framing to the artifact
        best_artifact.add_eval(self, max_evaluation, fr=framing)
        return best_artifact

    async def act(self):
        '''Agent acts by inventing new artifacts, considering methods, and learning from other examples
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
            
            #Read the invention method used from the artifact
            method_idx = artifact.obj[4]
            self.record_feedback(method_idx, their_opinion)
            
            #Consider new methods of inventing music using these lyrics as inspiration 
            self.reconsider_invention_methods(artifact.obj[0])
            
            #Also, if the other agent didn't like this one's artifact, try once more
            if their_opinion < 0.85:
                artifact = self.invent(self.n)
                
        #Now give the environment the final artifact for voting
        self.env.add_candidate(artifact)
        
    @aiomas.expose
    async def request_artifact_exchange(self, senders_artifact):
        '''Allows an agent to send an artifact to another agent in exchange
        for their opinion on the artifact, and an artifact of their own.
        
        :param artifact sender_artifact: The artifact the sender wishes to be evaluated
        :returns:  An opinion of received artifact, and an artifact for the requestor
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
    
    def record_feedback(self, method_idx, result):
        '''
        Allows an agent to be notified of feedback results so it can score its own
        invention methods
        :param int method_idx: The invention method index in its own invention_methods
        :param double vote_result: The voting result the artifact accumulated
        '''
        method_used = self.invention_methods[method_idx]
        times = method_used.times_utilized
        if (times == 0):
            avg = 0
        else:
            avg = method_used.average_rating
        times = times + 1
        self.invention_methods[method_idx].times_utilized = times
        self.invention_methods[method_idx].average_rating = (avg + result) / (times)