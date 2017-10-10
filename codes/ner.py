from nltk.tag import StanfordNERTagger
st = StanfordNERTagger('C:/Users/Ramanuja/Dropbox/Files/thesis_traffic/english.conll.4class.caseless.distsim.crf.ser.gz',
"C:/Users/Ramanuja/Dropbox/Files/thesis_traffic/stanford-english-core.jar") 
st.tag('Accident OB I-290 At north ave. Blocks 2 Left Lanes.'.split())
# replace @ -> at
# st - street
# Rd - Road
# U,u - you
#b/c - because
# Use that case study