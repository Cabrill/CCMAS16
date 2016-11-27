from pyknon.genmidi import Midi
from pyknon.music import NoteSeq

notes1 = NoteSeq("0 1 2 3 4 5 6 7 8 9 10 11")
midi = Midi(1, tempo=90)
midi.seq_notes(notes1, track=0)
midi.write("demo.mid")
