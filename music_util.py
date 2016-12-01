import pyknon, operator, numpy, random
from os import listdir
from os.path import isfile, join
from pyknon.music import NoteSeq


class MusicHelper:
    '''
    A class that contains methods for creating music from text, analyzing the music, and modifying it.
    '''
    #Hard-coded list of musical keys and their relative keys
    relative_keys = {
    'cmajor':'aminor',\
    'gmajor':'eminor',\
    'dmajor':'bminor',\
    'amajor':'fsharpminor',\
    'emajor':'csharpminor',\
    'bmajor':'gsharpminor',\
    'fsharpmajor':'dsharpminor',\
    'dflatmajor':'bflatminor',\
    'aflatmajor':'fminor',\
    'eflatmajor':'cminor',\
    'bflatmajor':'gminor',\
    'fmajor':'dminor',\
    'aminor':'cmajor',\
    'eminor':'gmajor',\
    'bminor':'dmajor',\
    'fsharpminor':'amajor',\
    'csharpminor':'emajor',\
    'gsharpminor':'bmajor',\
    'dsharpminor':'fsharpmajor',\
    'bflatminor':'dflatmajor',\
    'fminor':'aflatmajor',\
    'cminor':'eflatmajor',\
    'gminor':'bflatmajor',\
    'dminor':'fmajor'
    } 

    def __init__(self):
        self.music_keys = {}
        self.read_keys()
        
    def read_keys(self, mypath="keys/"):
        '''
        Initialization method that reads in all available keys, in a specified path.
        
        :param str mypath: The location of the keys to be read
        '''
        key_files = [f  for f in listdir(mypath) if isfile(join(mypath, f))]
        for key in key_files:
            new_key = key.replace(".txt", "")
            key_notes = NoteSeq("file://" + mypath + key)
            self.music_keys[new_key] = key_notes
    
    def determine_dominant_key(self, notes):
        '''
        Takes a list of notes and compares it to each key, then returns the key which the
        notes match best.
        
        :param str notes: A list of notes in string format or as a pyknon.music.NoteSeq.
        :returns: A string representation of the dominant key (i.e. 'fminor')
        '''
        if isinstance(notes, pyknon.music.NoteSeq):
            test_seq = notes
        else:
            if isinstance(notes, str):
                test_seq = NoteSeq(notes)
            elif isinstance(notes, list):
                test_seq = NoteSeq(''.join(notes))
            else:
                raise ValueError('Notes supplied to determine_dominant_key not in a known format (list, string, NoteSeq)')
    
        key_score = {}
        for key in self.music_keys.keys():
            key_score[key] = 0
        for note in test_seq:
            for key, value in self.music_keys.items():
                if note in value:
                    key_score[key] = key_score[key] + 1
        
        dominant_key = max(key_score, key=key_score.get)

        return dominant_key
    
    @staticmethod    
    def convert_phrase_to_notes(phrase):
        '''
        Takes a phrase and converts it to musical notes by extracting characters that represent
        musical notes.
        
        :param str phrase: Any string of text, to be converted to musical notes
        :returns: A list of notes extracted from the string.
        '''
        notes = []
        
        for word in phrase:
            for char in word:
                c = char.lower()
                if c == 'c':
                    notes.append("C")
                elif c == 'd':
                    notes.append("D")
                elif c == 'e':
                    notes.append("E")
                elif c == 'f':
                    notes.append("F")
                elif c == 'g':
                    notes.append("G")
                elif c == 'a':
                    notes.append("A")
                elif c == 'b':
                    notes.append("B")
                    
        return notes
    
    def conform_notes_to_key(self, notes, music_key):
        '''
        Takes a list of notes, and a music key and ensures that every note either
        belongs to that key or to its relative key.
        
        :param str notes: The notes to be matched
        :param str music_key: The name of the key to be matched (i.e. 'cmajor')
        :returns: A list of notes that are all in the specified key or its relative key.
        '''
        if isinstance(notes, NoteSeq):
            test_notes = notes
        else:
            test_notes = NoteSeq(' '.join(notes))

        new_note_list = []
        
        key_notes = self.music_keys[music_key]
        relative_key = self.relative_keys[music_key]
        
        if relative_key:
            relative_key_notes = self.music_keys[relative_key]
            
        for note in test_notes:
            if note in key_notes or note in relative_key_notes:
                new_note_list.append(note)

            else:
                new_note = numpy.random.choice(key_notes + relative_key_notes)
                new_note_list.append(new_note)
        
        return new_note_list
    
    @staticmethod
    def determine_theme(word):
        '''
        This method takes a word supplied by the agent and attempts to match items
        as a substring of themes encompassed by word lists.  If a matching theme
        is found then a list of MIDI instrument numbers is returned.
        *A substring will match (word=old, will_match=olden)
        
        :param str word: The word to use as a potential theme match
        :returns: A tuple containing tempo and a list of MIDI instruments in int format.
        '''
        #Filter out meaningless words
        if word in "and from say her him his she had the their for one":
            return None
        if word in "children kids young baby youth age kid bosom asleep nap quiet silence discovery learn education "\
                    "find taught teacher lesson boy girl":
            return (160,[76, 11]) #76-Pan Flute, 11-Music Box
            
        if word in "sky nature tree forest earth weather rain wind cloud drizzle mist atmosphere air bird "\
                    "tweet nest egg flight breasted morning sunrise day explore happy happiness calmly freshest":
            return (180, [100, 123]) #100-Atmosphere, 123-Seashore
        
        if word in "olden yearning ancients timely slowing pine wishing wistful lonely harkening past year month long ago "\
                    "true truly travel men night":
            return (140, [110, 90]) #110-Bagpipe, 90-Warm synth
            
        if word in "evil danger primal attacked attacking offense enemy dungeon wounded bleed blood murder"\
                    "death kill festering mortal peril endangered threatened ":
            return (300,[31, 86]) #31-Distortion Guitar, 86-Voice
        
        if word in "voices angels choir holy blessing prayer blessed fortunate fortune lucky benefit boon "\
                    "lucked chance godly heavenly golden happily peace above":
            return (180,[49, 53]) #49-String Ensemble, 53-Choir Ahs
            
        if word in "guitar solo rocked rocking tune played song strummed picked heard bass electric music":
            return (240, [26, 33]) #26-Steel guitar,  33-Acoustic bass
            
        if word in "native indigenous locally primitive folklore community social friendly smallest society "\
                    "farming garden landscape enjoyment enjoyed relaxed relaxation relaxing peaceful satisfied":
            return (160, [78, 116]) #78-Shakuhachi, 116-Woodblock
        
        if word == "random":
            random_tempo = random.randint(120,250)
            random_instr1 = random.randint(1, 127)
            random_instr2 = random.randint(1, 127)
            return (random.randint(60,180), [random_instr1, random_instr2])