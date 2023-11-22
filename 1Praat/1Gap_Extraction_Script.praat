project_dir$ = "/Users/hannahrowe/Google Drive/My Drive/Research/Acoustic Files/Analyzed/Text Grids (***)/"

# MAKE A LIST OF ALL TEXTGRIDS IN THE FOLDER
Create Strings as file list... list 'project_dir$'*.TextGrid
file_list = selected("Strings")
file_count = Get number of strings

# SETUP THE CSV FILE HEADER
writeFileLine: "/Users/hannahrowe/Desktop/Gap_Data_For_R.csv", "Participant, Task, Time, F1, F2, Duration"

# LOOP THROUGH THE LIST OF FILES
for current_file from 1 to file_count
	select Strings list
	file$ = Get string... current_file

	# READ IN THE TEXTGRID AND CORRESPONDING SOUND

	Read from file... 'project_dir$''file$'
	filename$ = selected$ ("TextGrid", 1)
	Read from file... 'project_dir$''filename$'.wav

	# FIND THE LABELED INTERVAL
	select TextGrid 'filename$'
	plus Sound 'filename$'

	# EXTRACT THE NAMES OF THE PRAAT OBJECTS
	thisSound$ = selected$("Sound")
	thisTextGrid$ = selected$("TextGrid")

	# EXTRACT THE NUMBER OF INTERVALS IN THE VOWEL TIER
	select TextGrid 'thisTextGrid$'
	numberOfPhonemes = Get number of intervals: 1

	# CREATE THE FORMANT OBJECT
	select Sound 'thisSound$'
	To Formant (burg)... 0 5 5000 0.01 50

	# LOOP THROUGH EACH INTERVAL IN THE VOWEL TIER
	for thisInterval from 1 to numberOfPhonemes

    	# GET THE NUMBER OF THE TIER
    	select TextGrid 'thisTextGrid$'
    	thisPhoneme$ = Get label of interval: 1, thisInterval
		
		if thisPhoneme$ != ""
    
    		# FIND THE DURATION
    		thisPhonemeStartTime = Get start point: 1, thisInterval
    		thisPhonemeEndTime = Get end point: 1, thisInterval
    		duration = thisPhonemeEndTime - thisPhonemeStartTime
			frameCount = duration/0.0025

			# EXTRACT FORMANT MEASUREMENTS
			for thisFrame from 1 to frameCount
				frameTime = thisPhonemeStartTime + (thisFrame * 0.0025)
				select Formant 'thisSound$'
   				f1 = Get value at time... 1 frameTime Hertz Linear
    			f2 = Get value at time... 2 frameTime Hertz Linear
				outputLine$ = filename$ + "," + thisPhoneme$ + "," + string$ (frameTime) + "," + string$ (f1) + "," + string$ (f2) + "," + string$ (duration)
				appendFileLine: "/Users/hannahrowe/Desktop/Gap_Data_For_R.csv", outputLine$
			endfor
		endif
	endfor
endfor
