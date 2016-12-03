import pyknon, operator, numpy, random, nltk
from os import listdir
from os.path import isfile, join
from pyknon.music import NoteSeq, Rest


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
        
    def derive_tracks_from_lyrics(self, lyrics, notes, track_duration):
        '''
        Attempts to find patterns in lyrics to be paired with the given notes
        to create new tracks of a specified duration
        '''
        num_notes = len(notes)
        lyrics_sentences = lyrics.replace("?", ".").replace("!", ".").split(".")
        #Remove the sentence token after the final .
        del lyrics_sentences[-1]
        num_sentences = len(lyrics_sentences)

        track_list = list()
        #Skip first sentence, as the Lead track will be based on it already
        #otherwise create a track for each sentence
        for i in range(1, num_sentences):
            #Count the words in this sentence
            lyric_words = nltk.word_tokenize(lyrics)
            num_words = len(lyric_words)
            #Count the number of characters in each word
            list_of_char_counts = list()
            for j in range(0,len(lyric_words)):
                list_of_char_counts.append(len(lyric_words[j]))
            num_chars_total = sum([cnt for cnt in list_of_char_counts])

            #Give notes equal time, in accordance with represented word length, plus time for rests
            duration_unit = (track_duration / (num_words  * num_chars_total)) - (num_words)
            
            #Every other track picks a pattern length differently
            if i % 2 == 0:
                pattern_length = list_of_char_counts[0]
                pattern_accent = list_of_char_counts[1]
                accent_occur = list_of_char_counts[2]
            else:
                pattern_length = list_of_char_counts[-1]
                pattern_accent = list_of_char_counts[-2]
                accent_occur = list_of_char_counts[-3]
            
            #Repeat the pattern equal to the number of words in the sentence
            this_track = NoteSeq()
            for n in range(0, num_words):
                for m in range(0, pattern_length):                    
                    note_to_append = m % num_notes
                    duration_count = m % list_of_char_counts[n]
                    note = notes[note_to_append]
                    note.duration = duration_count * duration_unit
                    if m % accent_occur == 3:
                        if m // accent_occur % 2 == 0:
                            note.transposition(pattern_accent)
                        else:
                            note.transposition(-pattern_accent)
                    this_track.append(note)
                #Rest for a second between tracks
                this_track.append(Rest(1))
            
            #Add the completed track
            track_list.append(this_track)
            
        return track_list
    
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
        if word in "and from say her him his she had the their for one you all":
            return None
        if word in "children kids young baby youth age kid bosom asleep nap quiet silence discovery learn education "\
                    "find taught teacher lesson boy girl":
            return (160,[75, 10, 15]) #76-Pan Flute, 11-Music Box, 16-Dulcimer
            
        if word in "sky nature tree forest earth weather rain wind cloud drizzle mist atmosphere air bird "\
                    "tweet nest egg flight breasted morning sunrise day explore happy happiness calmly freshest":
            return (180, [99, 122, 123]) #100-Atmosphere, 123-Seashore, 124-Bird Tweet
        
        if word in "olden yearning ancients timely slowing pine wishing wistful lonely harkening past year month long ago "\
                    "true truly travel men night":
            return (140, [109, 89, 117]) #110-Bagpipe, 90-Warm synth, 118-Melodic Drum
            
        if word in "evil danger primal attacked attacking offense enemy dungeon wounded bleed blood murder"\
                    "death kill festering mortal peril endangered threatened ":
            return (300,[30, 85, 114]) #31-Distortion Guitar, 86-Voice, 115-Steel drums
        
        if word in "voices angels choir holy blessing prayer blessed fortunate fortune lucky benefit boon "\
                    "lucked chance godly heavenly golden happily peace above":
            return (180,[48, 52, 98]) #49-String Ensemble, 53-Choir Ahs, 99-Crystal
            
        if word in "guitar solo rocked rocking tune played song strummed picked heard bass electric music":
            return (240, [25, 32, 117]) #26-Steel guitar,  33-Acoustic bass, 118-Melodic Tom
            
        if word in "native indigenous locally primitive folklore community social friendly smallest society "\
                    "farming garden landscape enjoyment enjoyed relaxed relaxation relaxing peaceful satisfied":
            return (160, [78, 110, 116]) #79-Whistle, 110-Fiddle, 117-Taiko Drum
        
        if word == "random":
            random_tempo = random.randint(120,250)
            random_instr1 = random.randint(1, 127)
            random_instr2 = random.randint(1, 127)
            random_instr3 = random.randint(1, 127)
            return (random.randint(60,180), [random_instr1, random_instr2, random_instr3])

    @staticmethod  
    def determine_instrument(midi_int):
        '''
        Static method that identifies a category and instrument for a given int representing a MIDI instrument.
        
        :param int midi_int: The int of the MIDI instrument
        :returns: A tuple containing the category and instrument name for the given int.
        '''
        #Add one because our MIDI writer uses 0-based index
        midi_int = midi_int + 1
        
        #Set the default
        category = 'None'
        instrument = 'None'
        
        if  midi_int < 1 or midi_int > 128:
            return type_instrument
        
        if midi_int <= 8: 
            category = 'Piano'
            instrument = {
                1 : 'Acoustic Grand Piano',
                2 : 'Bright Acoustic Piano',
                3 : 'Electric Grand Piano',
                4 : 'Honky-tonk Piano',
                5 : 'Electric Piano 1',
                6 : 'Electric Piano 2',
                7 : 'Harpsichord',
                8 : 'Clavinet'
            }[midi_int]
        elif midi_int <= 16:
            category = 'Chromatic Percussion'
            instrument = {
                9  : 'Celesta',
                10 : 'Glockenspiel',
                11 : 'Music Box',
                12 : 'Vibraphone',
                13 : 'Marimba',
                14 : 'Xylophone',
                15 : 'Tubular Bells',
                16 : 'Dulcimer'
            }[midi_int]

        elif midi_int <= 24:
            category = 'Organ'
            instrument = {
                17 : 'Drawbar Organ',
                18 : 'Percussive Organ',
                19 : 'Rock Organ',
                20 : 'Church Organ',
                21 : 'Reed Organ',
                22 : 'Accordion',
                23 : 'Harmonica',
                24 : '*Tango Accordion'
            }[midi_int]

        elif midi_int <= 32:
            category = 'Guitar'
            instrument = {
                25 : 'Acoustic Guitar (nylon)',
                26 : 'Acoustic Guitar (steel)',
                27 : 'Electric Guitar (jazz)',
                28 : 'Electric Guitar (clean)',
                29 : 'Electric Guitar (muted)',
                30 : 'Overdriven Guitar',
                31 : 'Distortion Guitar',
                32 : 'Guitar harmonics'
            }[midi_int]

        elif midi_int <= 40:
            category = 'Bass'
            instrument = {
                33 : 'Acoustic Bass',
                34 : 'Electric Bass (finger)',
                35 : 'Electric Bass (pick)',
                36 : 'Fretless Bass',
                37 : 'Slap Bass 1',
                38 : 'Slap Bass 2',
                39 : 'Synth Bass 1',
                40 : 'Synth Bass 2'
            }[midi_int]
            
        elif midi_int <= 52:
            category = 'Strings'
            instrument = {        
                41 : 'Violin',
                42 : 'Viola',
                43 : 'Cello',
                44 : 'Contrabass',
                45 : 'Tremolo Strings',
                46 : 'Pizzicato Strings',
                47 : 'Orchestral Harp',
                48 : 'Timpani',
                49 : 'String Ensemble 1',
                50 : 'String Ensemble 2',
                51 : 'Synth Strings 1',
                52 : 'Synth Strings 2'
            }[midi_int]
        
        elif midi_int <= 56:
            category = 'Voice'
            instrument = {   
                53 : 'Choir Aahs',
                54 : 'Voice Oohs',
                55 : 'Synth Voice',
                56 : 'Orchestra Hit'
            }[midi_int]

        elif midi_int <= 64:
            category = 'Brass'
            instrument = {  
                57 : 'Trumpet',
                58 : 'Trombone',
                59 : 'Tuba',
                60 : 'Muted Trumpet',
                61 : 'French Horn',
                62 : 'Brass Section',
                63 : 'Synth Brass 1',
                64 : 'Synth Brass 2'
            }[midi_int]


        elif midi_int <= 72:
            category = 'Reed'
            instrument = {  
                65 : 'Soprano Sax',
                66 : 'Alto Sax',
                67 : 'Tenor Sax',
                68 : 'Baritone Sax',
                69 : 'Oboe',
                70 : 'English Horn',
                71 : 'Bassoon',
                72 : 'Clarinet'
            }[midi_int]

        elif midi_int <= 80:
            category = 'Pipe'
            instrument = {  
                73 : 'Piccolo',
                74 : 'Flute',
                75 : 'Recorder',
                76 : 'Pan Flute',
                77 : 'Blown Bottle',
                78 : 'Shakuhachi',
                79 : 'Whistle',
                80 : 'Ocarina'
            }[midi_int]

        elif midi_int <= 88:
            category = 'Synth Lead'
            instrument = { 
                81 : 'Square',
                82 : 'Sawtooth',
                83 : 'Calliope',
                84 : 'Chiff',
                85 : 'Charang',
                86 : 'Voice',
                87 : 'Fifths',
                88 : '(Bass + Lead)'
            }[midi_int]

        elif midi_int <= 96:
            category = 'Synth Pad'
            instrument = { 
                89 : 'New age',
                90 : 'Warm',
                91 : 'Polysynth',
                92 : 'Choir',
                93 : 'Bowed',
                94 : 'Metallic',
                95 : 'Halo',
                96 : 'Sweep'
            }[midi_int]

        elif midi_int <= 104:
            category = 'Synth Effects'
            instrument = { 
                97 : 'Rain',
                98 : 'Soundtrack',
                99 : 'Crystal',
                100 : 'Atmosphere',
                101 : 'Brightness',
                102 : 'Goblins',
                103 : 'Echoes',
                104 : 'Sci-fi'
            }[midi_int]

        elif midi_int <= 112:
            category = 'Ethnic'
            instrument = { 
                105 : 'Sitar',
                106 : 'Banjo',
                107 : 'Shamisen',
                108 : 'Koto',
                109 : 'Kalimba',
                110 : 'Bag pipe',
                111 : 'Fiddle',
                112 : 'Shanai'
            }[midi_int]
        
        elif midi_int <= 119:
            category = 'Percussive'
            instrument = { 
                113 : 'Tinkle Bell',
                114 : 'Agogo',
                115 : 'Steel Drums',
                116 : 'Woodblock',
                117 : 'Taiko Drum',
                118 : 'Melodic Tom',
                119 : 'Synth Drum'
            }[midi_int]

        else:
            category = 'Sound Effect'
            instrument = { 
                120 : 'Reverse Cymbal',
                121 : 'Guitar Fret Noise',
                122 : 'Breath Noise',
                123 : 'Seashore',
                124 : 'Bird Tweet',
                125 : 'Telephone Ring',
                126 : 'Helicopter',
                127 : 'Applause',
                128 : 'Gunshot'
            }[midi_int]
        
        return (category, instrument)