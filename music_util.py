import pyknon
import operator
import numpy
from os import listdir
from os.path import isfile, join
from pyknon.music import NoteSeq


class MusicHelper:
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
        key_files = [f  for f in listdir(mypath) if isfile(join(mypath, f))]
        for key in key_files:
            new_key = key.replace(".txt", "")
            key_notes = NoteSeq("file://" + mypath + key)
            self.music_keys[new_key] = key_notes
    
    def determine_dominant_key(self, notes):
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
        
    def convert_phrase_to_notes(self, phrase):
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