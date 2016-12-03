import aiomas, re, random, nltk
from creamas.core import CreativeAgent, Environment, Simulation, Artifact
from mc import markov_chain, generate, sanitize, determineOrder, likelihood, format_for_printing
from list_mem import ListMemory
from pyknon.music import NoteSeq, Rest

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

class MusicAgent(CreativeAgent):
    '''A sample agent implementation.

    Agent invents new words be generating them at random and evaluating them
    with respect to its own vocabulary. 

    Agent learns its vocabulary from the file given at initialization.
    '''

    def __init__(self, env, mcprobs, mcstates, helper, mem_len, service_addr, encoding='utf8', n=20,
                 wlen_limits=(3,10), chars='abcdefghijklmnopqrstuvwxyz'):
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
        lyrics = ""
        for i in range(0,3):
            text_length = random.randint(*self.wlen_limits)
            lyrics = lyrics + " " + format_for_printing(generate(self.MarkovChainProbs, text_length))
        lyrics = lyrics.strip()
        
        word_theme, music_theme, track_list = self.create_music(lyrics)
        
        tuple_artifact = (lyrics, word_theme, music_theme, track_list)
        
        return Artifact(self, tuple_artifact, domain=tuple)
        
    def create_music(self, lyrics):
        '''Generate music from text.

        Music is generated by deriving notes from text, harmonizing those notes, and splitting them
        into separate tracks, choosing a theme based on the text and then matching the tracks up.

        :returns: word_theme=a string representing inspiration, music_theme=tempo and instrument list, 
        track_list=list of tracks composed of notes
        '''
        derived_notes = self.music_helper.convert_phrase_to_notes(lyrics)
        music_key = self.music_helper.determine_dominant_key(derived_notes)
        notes = self.music_helper.conform_notes_to_key(derived_notes, music_key)
       
        lyric_words = nltk.word_tokenize(lyrics)
        phrase_len = len(lyric_words)
        note_len = len(notes)
        
        track_list = []
        lead_track = NoteSeq()
        other_notes = NoteSeq()
        
        #Separate notes into lead track/others, assign word-based duration
        for i in range(0,note_len):
            #Associate each note for the lead track with a word in the lyrics,
            #until we're out of words, then put the rest in "other"
            if i < phrase_len:
                word_ptr = i
                word_for_note = lyric_words[word_ptr]
                
                #Set notes duration based on word length
                notes[i].dur = len(word_for_note) * 0.25
                lead_track.append(notes[i])
            else:
                other_notes.append(notes[i])
                
        #Insert rests for lead track at punctuation marks
        rest_count = 0
        for i in range(0, len(lead_track)):
            if lyric_words[i] in (',', '.', ';', '!', '?', '"', ':', '/', '\\'):
                lead_track.insert(i+rest_count, Rest(2))
                rest_count = rest_count + 1
        
        #See how long the lead track is, then add it
        lead_track_duration = sum([note.dur for note in lead_track])      
        track_list.append(lead_track)
        
        #Now attempt to detect patterns in the lyrics in combination with the
        #other notes, for the purpose of creating more tracks
        if len(other_notes) > 3:
            pattern_tracks = self.music_helper.derive_tracks_from_lyrics(lyrics, other_notes, lead_track_duration)
            for i in range(0, len(pattern_tracks)):
                track_list.append(pattern_tracks[i])
                
        #If there weren't enough notes for a second track, make a small accompaniment instead
        if len(track_list) < 2:
            #Determine an ideal duration pattern length
            num_notes = len(other_notes)
            
            #If there aren't enough notes, copy some from the lead track
            if num_notes < 4:
                lead_length = len(lead_track)
                for i in range(lead_length-1,lead_length-8, -1):
                    other_notes.append(lead_track[i])
                num_notes = len(other_notes)
                
            pattern_length = 8
            for potential_length in range(pattern_length, 1, -1):
                if num_notes % potential_length == 0:
                    break

            #Don't allow pattern length to exceed phrase word length
            pattern_length = min(potential_length, phrase_len)

            pattern_list = list()
            #Take the length of the words in the end of the phrase
            for p in range(phrase_len-1, phrase_len-pattern_length-1, -1):
                word = lyric_words[p]
                pattern_list.append(len(word))

            pattern_sum = sum(int(x) for x in pattern_list)
            num_pattern_in_song = num_notes / len(pattern_list)
            pattern_dur = lead_track_duration / num_pattern_in_song
            dur_value = pattern_dur / pattern_sum

            #Assign note length based on derived word lengths
            for j in range(0, num_notes):
                pat_idx = i % pattern_length
                other_notes.dur = pattern_list[pat_idx] * dur_value
                
            track_list.append(other_notes)
                
        #Now make the tracks equal in duration, so there isn't long silence
        longest_duration = lead_track_duration
        for i in range(1, len(track_list)):
            this_track_duration = sum([note.dur for note in track_list[i]])
            if this_track_duration > longest_duration:
                longest_duration = this_track_duration
                
        for i in range(0, len(track_list)):
            #Calculate this track duration
            this_track_duration = sum([note.dur for note in track_list[i]])
            #Add some rests before/during to make it centered
            if this_track_duration < longest_duration:
                insert_rest = (longest_duration - this_track_duration) / 2
                track_list[i].insert(0,Rest(insert_rest))
                track_list[i].append(Rest(insert_rest))
            
            #Add a 1 second pause to the end of the longest track so it ends gracefully
            if this_track_duration == longest_duration:
                track_list[i].insert(0,Rest(1))
        
        #Find a word that will provide a theme for the song
        music_theme = None
        for i in range(0, phrase_len):
            if music_theme != None:
                break
            word_theme = lyric_words[i]
            if len(word_theme) > 2:
                music_theme = self.music_helper.determine_theme(word_theme)
        #No matching words were found, choose one at random
        if music_theme == None:
            word_theme = "random"
            music_theme = self.music_helper.determine_theme(word_theme)
            
        return word_theme, music_theme, track_list

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
        