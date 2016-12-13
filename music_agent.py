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
    :returns: The number of edits required to make one string match the other
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
    A list of functions used to generate parameters in the creation of music, 
    and minimal statistics about the overall invention method
    '''
    def __init__(self, method_list, name):
        '''        
        :param list method_list: A list of methods accepting a string, used for invention
        '''
        self.times_utilized = 0
        self.average_rating = 1
        self.method_list = method_list
        self.name = name

class MusicAgent(CreativeAgent):
    '''An agent that creates music in the form of lyrics and instrument tracks.

    Agent invents new lyrics based on their Markov Chain probabilities computed
    from inspecting their assigned TXT from the inspiration set at initialization.
    Which is then modified when learning from new artifacts created by other agents.
    '''

    def __init__(self, env, mcprobs, mcstates, helper, mem_len, service_addr, n=20,
                 wlen_limits=(10,15), method_limits = 5):
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
                    
        # Tokenize the text into sentences.
        sentences = nltk.sent_tokenize(raw_text)

        # Tokenize each sentence to words. Each item in 'words' is a list with
        # tokenized words from that list.
        tokenized_sentences = []
        for s in sentences:
            w = nltk.word_tokenize(s)
            tokenized_sentences.append(w)

        # Sanitize the tokens of each sentence
        tokenized_sentences = [sanitize(sentence) for sentence in tokenized_sentences]
        
        # Use the same order as existing learned Markov Chain
        order = self.mcpOrder

        # Now we are ready to create the state transitions
        for data in enumerate(tokenized_sentences):
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
        
    def evaluate(self, artifact):
        '''Evaluate given artifact by checking its inherent value and novelty.
        
        :param artifact: :class:`~creamas.core.Artifact` to be evaluated
        :returns:
            (evaluation, word)-tuple, containing both the evaluation and the
            evaluation method
        '''
        lyric_eval = self.eval_lyrics(artifact)
        music_eval = self.eval_music(artifact)
        
        #Give greater weight to music eval than lyrics
        overall_eval = (lyric_eval + music_eval*3) / 4
        
        framing = {"Overall": overall_eval, 
            "Lyric": lyric_eval,
            "Music": music_eval}
        
        return overall_eval, framing
        
    def eval_lyrics(self, artifact):
        '''Evaluates an artifact purely on the basis of the lyrics it contains, not paying attention to the music.
    
        :param artifact: :class:`~creamas.core.Artifact` containing the lyrics to be evaluated
        :returns: float value from 0 to 1 representing the quality of lyrics
        '''
        grammar_chk = self.eval_lyrics_grammar(artifact)
        
        novelty_score = self.eval_lyrics_novelty(artifact)
        prob_score = self.eval_lyrics_probability(artifact)
        length_score = self.eval_lyrics_length(artifact)
       
        #Give greater evaluation to longer lyrics than to novelty or probability
        evaluation = grammar_chk * (prob_score + novelty_score + length_score*5) / 7
        return evaluation
    
    def eval_lyrics_length(self, artifact):
        '''Evaluates the artifact on the basis of the length of the lyrics, assuming longer lyrics are better for
        the purpose of generating music.
        
        :param artifact: :class:`~creamas.core.Artifact` containing the lyrics to be evaluated
        :returns: float value from 0 to 1 representing the length of lyrics
        '''
        #Points for length diminishes as it increases, never passing 1
        length_eval = 0
        len_val = 1
        for word in nltk.word_tokenize(artifact.obj[0]):
            len_val = len_val / 2
            length_eval = length_eval + len_val
        
        return length_eval
        
    def eval_lyrics_probability(self, artifact):
        '''Evaluates the probability of the lyrics having occured, based off the agent's own Markov Chain.
        
        :param artifact: :class:`~creamas.core.Artifact` containing the lyrics to be evaluated
        :returns: float value from 0 to 1 representing the probability of the lyrics occurring
        '''
        probability = 1 - likelihood(artifact.obj[0], self.MarkovChainProbs)
        return probability
        
    def eval_lyrics_grammar(self, artifact):
        '''This is meant to be a simple way to exclude artifacts that
        end in a word known to be invalid ending for a sentence.
        
        :param artifact: :class:`~creamas.core.Artifact` containing the lyrics to be evaluated
        :returns: A score of the artifact's employed grammar, from 0 to 1.
        '''
        grammar_score = 1
        words = nltk.word_tokenize(artifact.obj[0])
        if ''.join(words[-1:]) in ("and", "the", "of", "if", "their", "a", "as", "but", "or" ):
            grammar_score = 0
        return grammar_score
        
    def eval_lyrics_novelty(self, artifact):
        '''Assign a value for the novelty of an artifact.  This value is based one
        the similarity of the artifact to previously recorded artifacts and
        therefore represents how unique the artifact is to this agent.   
        
        :param artifact: :class:`~creamas.core.Artifact` to be evaluated
        :returns:  A score of the lyrics novelty, from 0 to 1
        '''
        # We will choose that the novelty is maximal if agent's memory is empty.

        if len(self.mem.artifacts) == 0:
            return 1.0

        novelty = 1.0
        evaluation_lyrics = artifact.obj[0]
        matching_lyrics = self.mem.artifacts[0].obj[0]
        for memart in self.mem.artifacts:
            lyrics = memart.obj[0]
            lev = levenshtein(evaluation_lyrics, lyrics)
            mlen = max(len(evaluation_lyrics), float(len(lyrics)))
            current_novelty = float(lev) / mlen
            if current_novelty < novelty:
                novelty = current_novelty
                matching_lyrics = lyrics
                
        return novelty

    def eval_music(self,artifact):
        '''
        Attempts to evaluate the quality of music by looking at instrument selection,
        pattern repetition (melody) and note harmony (adherence to a key).
        
        :param artifact: :class:`~creamas.core.Artifact` containing the music to be evaluated
        :returns: float value from 0 to 1 representing the quality of music
        '''
        #evaluate instruments
        instrument_eval = self.eval_music_instruments(artifact)
        melody_eval = self.eval_music_melody(artifact)
        harmony_eval = self.eval_music_harmony(artifact)

        #Give higher weight to instrument choice and harmony than to melody
        eval_score = (melody_eval + instrument_eval*3 + harmony_eval*2) / 6
        
        return eval_score
        
    def eval_music_instruments(self, artifact):
        '''
        Evaluates the choice of instruments in the music, based on the assumption that the first
        track will lead the song, the second track will provide rhythm and the third track will
        provide a beat.
    
        :param artifact: :class:`~creamas.core.Artifact` containing the music to be evaluated
        :returns: float value from 0 to 1 representing the quality of instrument selection
        '''
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
            
        inst_eval = max(0,inst_eval / 9) #Divide by max possible, with a floor of zero
        
        return inst_eval
        
    def eval_music_melody(self, artifact):
        '''
        Evaluates the melody of the music, based on the presence of patterns and how often each
        pattern is repeated.
    
        :param artifact: :class:`~creamas.core.Artifact` containing the music to be evaluated
        :returns: float value from 0 to 1 representing the quality of music's melody
        '''
        minPhrase = 12 #min notes that a phrase should have
        notes = self.music_helper.convert_tracks_to_string(artifact.obj[3]) #only the notes of track_list
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
        melody_eval = 0
        ideal_count = 5
        for i in range(min(4, max(0,len(sorted_occ)-1)), 0,  -1):
            if len(sorted_occ[i]) > 1:
                phrase_count = sorted_occ[i][1]
                phrase_score = max(0,(ideal_count - abs(phrase_count - ideal_count)) / ideal_count) #Measures proximity to ideal
                phrase_score = phrase_score * (0.0715 * (i+1))#This 0.0715*i gives depreceating returns for more patterns
                melody_eval = melody_eval + phrase_score
        
        melody_eval = min(1, melody_eval) #Correct rounding error that would result in 1.01    
        
        return melody_eval
        
    def eval_music_harmony(self, artifact):
        '''
        Evaluates the harmony of the music, by identifying the dominant musical key in the notes,
        then creating a version in which all the notes adhere to that key, and then comparing
        the difference between the original note list and the note list that conforms to the key.
    
        :param artifact: :class:`~creamas.core.Artifact` containing the music to be evaluated
        :returns: float value from 0 to 1 representing the quality of music's harmony
        '''
        notes = self.music_helper.convert_tracks_to_string(artifact.obj[3]) #only the notes of track_list
        
        #Evaluate note harmony by comparing the notes to the ideal
        dom_key = self.music_helper.determine_dominant_key(notes)
        perfect_notes = self.music_helper.convert_tracks_to_string((self.music_helper.conform_notes_to_key(notes, dom_key)))
        
        #Harmony is equal to the distance from the actual notes to the notes in perfect harmony
        lev_dist = levenshtein(notes, perfect_notes)
        mlen = max(len(notes), len(perfect_notes))
        lev_dist = lev_dist/mlen
        harmony_eval = 1 - lev_dist
        
        return harmony_eval

    def generate_lyrics(self):
        '''Generate new text.

        Text is generated by state transition probability of a markov chain. Text length is
        in ``wlen_limits``.

        :returns: a string of text wrapped as :class:`~creamas.core.artifact.Artifact`
        '''
        text_length = random.randint(*self.wlen_limits)
        lyrics = generate(self.MarkovChainProbs, text_length)

        #Create an artifact containing the creations
        tuple_artifact = (lyrics, None)
        
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
        word_theme = random.choice([word for word in lyric_words if len(word) > 4])
        if word_theme == None:
            self.music_helper.determine_theme("random")
        else:
            tempo = max(120,invention_method.method_list[0](word_theme) * invention_method.method_list[4](word_theme) % 600)
            instr1 = (invention_method.method_list[0](word_theme) * invention_method.method_list[3](word_theme)) % 127
            instr2 = (invention_method.method_list[1](word_theme) * invention_method.method_list[2](word_theme)) % 127
            instr3 = (invention_method.method_list[4](word_theme) * invention_method.method_list[0](word_theme)) % 127
            music_theme = (tempo, [instr1, instr2, instr3])

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
            #Use the probability of a sample of the lyrics as a seed for creation
            lyric_sample = ' '.join(lyrics_tokens[0:self.mcpOrder+1])
            lyric_seed = 1 - likelihood(lyric_sample, self.MarkovChainProbs)
            best_method = None
            self.invention_methods.sort(key=lambda method_list: method_list.average_rating, reverse=True)
            for invention_method in self.invention_methods:
                if invention_method.average_rating >= lyric_seed:
                    best_method = invention_method
                    break
            #If no method chosen based on the lyrics, create a new one or choose the highest ranked one 
            if not best_method:
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
        name = ""
        method_list = []
        lyrics_tokens = nltk.word_tokenize(lyrics)
        token_len = len(lyrics_tokens)
        #Create five methods used to derive values used in music generation
        for method_ref in range(0, 5):
            #Prevent a high order from causing an out-of-index on the lyrics
            lyric_index = min(method_ref, token_len-method_ref-self.mcpOrder-2)
            #Use the probability of a sample of the lyrics as a seed for creation
            lyric_sample = ' '.join(lyrics_tokens[lyric_index:lyric_index+self.mcpOrder+1])
            lyric_seed = 1 - likelihood(lyric_sample, self.MarkovChainProbs)
            if lyric_seed >= 0.9:
                new_method = lambda text: max(1,len(text) % ((method_ref+1)*5))
            elif lyric_seed >= 0.8:
                new_method = lambda text: max(1,ord(text[0]) % (1+(43*method_ref)))
            elif lyric_seed >= 0.7:
                new_method = lambda text: max(1,ord(text[-1]) % (1+(43*method_ref)))
            elif lyric_seed >= 0.6:
                new_method = lambda text: max(1,ord(text[0]) + ord(text[-1]) % (1+(43*method_ref)))
            elif lyric_seed >= 0.5:
                new_method = lambda text: max(1,len(text) % max(3,(ord(text[0])%(1+(43*method_ref)))))
            elif lyric_seed >= 0.4:
                new_method = lambda text: max(1,sum([ord(c) for c in text]) % (1+(43*method_ref)))
            elif lyric_seed >= 0.3:
                new_method = lambda text: max(1,abs(ord(text[0])-ord(text[1]) % ((method_ref+1)*5)))
            elif lyric_seed >= 0.2:
                new_method = lambda text: max(1,abs(ord(text[0])-ord(text[-1]) % ((method_ref+1)*5)))
            elif lyric_seed >= 0.1:
                new_method = lambda text: max(1,abs(len(text)% 1 + abs(ord(text[0])- 43 - method_ref)))
            else:
                new_method = lambda text: max(1,abs(len(text)% 1 + abs(ord(text[-1])-43- method_ref)))
        
            method_list.append(new_method)
            name = name + lyric_sample[0] + "{0:.3f}".format(lyric_seed) + lyric_sample[-1]
        
        return invention_method(method_list, name)
        
    def replace_invention_method(self, new_method):
        '''Allows an agent to replace one of their invention methods with a new one
        
        :param invention_method new_method:  The new invention method to be added to agent's methods
        '''
        self.invention_methods.sort(key=lambda method_list: method_list.average_rating, reverse=True)
        
        #Remove the worst performing invention method, if necessary
        if len(self.invention_methods) == self.method_limit:
            del self.invention_methods[-1]
            
        self.invention_methods.append(new_method)
        
    def invent(self, n=10):
        '''Invents a new song

        Generates multiple words and selects the lyrics sentence with the highest rating.

        :param int n: Number of sentences to consider
        :returns:
            a sentence wrapped as :class:`~creamas.core.artifact.Artifact` and its
            evaluation.
        '''
        # Before we generate a new artifact, we should incorporate our learning
        # by recomputing the Markov Chain probabilities
        self.compute_probabilities()
        
        lyrics = ""
        for i in range(0,3):
            best_lyrics = self.generate_lyrics()
            max_evaluation = self.eval_lyrics(best_lyrics)
            for _ in range(n-1):
                new_lyrics = self.generate_lyrics()
                new_eval = self.eval_lyrics(new_lyrics)
                if new_eval > max_evaluation:
                    best_lyrics = new_lyrics
                    max_evaluation = new_eval
            lyrics = lyrics + " " + format_for_printing(best_lyrics.obj[0])
        
        #Remove any leading or trailing spaces
        lyrics = lyrics.strip()
        
        #Choose an invention method for the song and record its index for later evaluation
        invention_method = self.choose_invention_method(lyrics)
        method_name = invention_method.name
        
        #Create a word theme, music theme, and track_list
        word_theme, music_theme, track_list = self.create_music(lyrics, invention_method)
        
        #Create an artifact containing the creations
        artifact = Artifact(self, (lyrics, word_theme, music_theme, track_list, method_name), domain=tuple)
        personal_eval, framing = self.evaluate(artifact)
        
        # Add evaluation and framing to the artifact
        artifact.add_eval(self, artifact, fr=framing)
        
        return artifact

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
            their_opinion_lyrics, their_opinion_music, requested_artifact = await random_agent.request_artifact_exchange(artifact)
            their_overall_opinion = (their_opinion_lyrics + their_opinion_music) / 2
            
            #Make an opinion on their artifact's lyrics
            my_opinion_lyrics = self.eval_lyrics(requested_artifact)
            my_opinion_music = self.eval_music(requested_artifact)
        
            #If this agent liked the returned artifact's lyrics, then learn from them
            if my_opinion_lyrics > 0.85:
                self.mem.memorize(requested_artifact)
                self.learn(requested_artifact)
            
            #Read the invention method used from the artifact
            method_name = artifact.obj[4]
            self.record_feedback(method_name, their_opinion_music)
            
            #Also, if the other agent didn't like this one's artifact, try once more
            if their_overall_opinion < 0.85:
                #Consider new methods of inventing music using these lyrics as inspiration 
                self.reconsider_invention_methods(artifact.obj[0])
                artifact = self.invent(self.n)
                                                
                #if this agent like the returned artifact's music, ask for their invention method
                if my_opinion_music > 0.8:
                    new_method = await random_agent.request_invention_method(artifact)
                    if new_method:
                        self.replace_invention_method(new_method)
                
        #Now give the environment the final artifact for voting
        self.env.add_candidate(artifact)
        
    @aiomas.expose
    async def request_artifact_exchange(self, senders_artifact):
        '''Allows an agent to send an artifact to another agent in exchange
        for their opinion on the artifact, and an artifact of their own.
        
        :param artifact sender_artifact: The artifact the sender wishes to be evaluated
        :returns:  An opinion on the music of received artifact, and an artifact for the requestor
        '''
        my_opinion_of_lyrics = self.eval_lyrics(senders_artifact)
        my_opinion_of_music = self.eval_music(senders_artifact)
        
        #Learn from this artifact if it's liked and return the favor
        if my_opinion_of_lyrics > 0.85:
            self.mem.memorize(senders_artifact)
            self.learn(senders_artifact)
            my_artifact = self.invent()
        else:
            #If their artifact was not good, give them something random from memory
            #if one is available, and if it's not then create one.
            if len(self.mem.artifacts) > 0:
                my_artifact = random.choice(self.mem.artifacts)
            else:
                my_artifact = self.invent()
        
        return my_opinion_of_lyrics, my_opinion_of_music, my_artifact

    @aiomas.expose
    async def request_invention_method(self, senders_artifact):
        '''Allows an agent to request they share their best invention method with
        them.
        
        :param artifact sender_artifact: An artifact from the sender as a gift
        :returns:  The agents invention method they wish to share
        '''
        shared_invention_method = None
        shareable_methods = [method for method in self.invention_methods if method.times_utilized >= 5]
        
        #Only consider sharing if any have been used enough to warrant sharing
        if len(shareable_methods) > 0:
            #Sort invention methods so the best are at the top
            shareable_methods = shareable_methods.sort(key=lambda method_list: method_list.average_rating, reverse=True)
            
            my_opinion_of_music = self.eval_music(senders_artifact)
            #Not a very good artifact, sender needs all the help they can get
            if my_opinion_of_music < 0.6:
                shared_invention_method = shareable_methods[0]
            else:#Give them the agent's worst method, to humor them
                shared_invention_method = shareable_methods[-1]
                
        return shared_invention_method
        
        
    def record_feedback(self, method_name, result):
        '''
        Allows an agent to be notified of feedback results so it can score its own
        invention methods
        :param int method_idx: The invention method index in its own invention_methods
        :param float vote_result: The voting result the artifact accumulated
        '''
        method_idx = [self.invention_methods.index(im) for im in self.invention_methods if im.name ==method_name][0]
        method_used = self.invention_methods[method_idx]
        
        times = method_used.times_utilized
        if (times == 0):
            avg = 0
        else:
            avg = method_used.average_rating
        times = times + 1
        self.invention_methods[method_idx].times_utilized = times
        self.invention_methods[method_idx].average_rating = (avg + result) / (times)