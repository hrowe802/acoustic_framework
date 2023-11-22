project_dir$ = "/Users/hannahrowe/Google Drive/My Drive/Research/Acoustic Files/Analyzed/Text Grids (***)/"

# MAKE A LIST OF ALL TEXTGRIDS IN THE FOLDER
Create Strings as file list... list 'project_dir$'*.TextGrid
file_list = selected("Strings")
file_count = Get number of strings

# SETUP THE CSV FILE HEADER
writeFileLine: "/Users/hannahrowe/Desktop/Cepstrum_Data_For_Python.csv", "Participant, Task, CPPS"

# LOOP THROUGH THE LIST OF FILES
for current_file from 1 to file_count
	select Strings list
	file$ = Get string... current_file

	# READ IN THE TEXTGRID & CORRESPONDING SOUND
	Read from file... 'project_dir$''file$'
	filename$ = selected$ ("TextGrid", 1)
	Read from file... 'project_dir$''filename$'.wav

	# FIND THE LABELED INTERVAL
	select TextGrid 'filename$'
	plus Sound 'filename$'

	# EXTRACT THE NAMES OF PRAAT OBJECTS
	thisSound$ = selected$("Sound")
	thisTextGrid$ = selected$("TextGrid")

	# EXTRACT THE NUMBER OF INTERVALS IN THE FULL TIER
	select TextGrid 'thisTextGrid$'
	numberOfPhonemes = Get number of intervals: 3

	# CREATE THE FORMANT OBJECT
	select Sound 'thisSound$'
	To Formant (burg)... 0 5 5000 0.0025 50

	# LOOP THROUGH EACH INTERVAL ON THE FULL TIER
	for thisInterval from 1 to numberOfPhonemes

    	# GET THE LABEL OF THE INTERVAL
    	select TextGrid 'thisTextGrid$'
    	thisPhoneme$ = Get label of interval: 3, thisInterval
		
		if thisPhoneme$ != ""
    
    		# EXTRACT CEPSTRAL INFO
			thisPhonemeStartTime = Get start point: 3, thisInterval
    		thisPhonemeEndTime = Get end point: 3, thisInterval
			select Sound 'filename$'
			call cepstralInfo 'thisPhonemeStartTime' 'thisPhonemeEndTime'
			outputLine$ = filename$ + "," + thisPhoneme$ + "," + string$ (cpps)
			appendFileLine: "/Users/hannahrowe/Desktop/Cepstrum_Data_For_Python.csv", outputLine$
		endif
	endfor
endfor

procedure cepstralInfo .onset .offset
	# SET WINDOW LENGTH
	Extract part... '.onset' '.offset' Hamming 1 no
	snd = selected ("Sound")

	# EXTRACT CPPS
	To PowerCepstrogram: 60, 0.002, 5000, 50
	Smooth: 0.02, 0.0005
	cpps = Get CPPS: "yes", 0.02, 0.0005, 60, 330, 0.05, "Parabolic", 0.001, 0, "Exponential decay", "Robust"
	Remove
endproc