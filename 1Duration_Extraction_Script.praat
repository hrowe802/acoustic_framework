project_dir$ = "/Users/hannahrowe/Google Drive/My Drive/Research/Acoustic Files/Analyzed/Text Grids (***)/"

# MAKE A LIST OF ALL TEXTGRIDS IN THE FOLDER
Create Strings as file list... list 'project_dir$'*.TextGrid
file_list = selected("Strings")
file_count = Get number of strings

# SETUP THE CSV FILE HEADER
writeFileLine: "/Users/hannahrowe/Desktop/Duration_Data_For_R.csv", "Participant, Task, Duration, DDKRate"

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
	numberOfPhonemes = Get number of intervals: 4
	# (3 if by rep, 4 if by full)

	# CREATE THE FORMANT OBJECT
	select Sound 'thisSound$'
	To Formant (burg)... 0 5 5000 0.0025 50

	# LOOP THROUGH EACH INTERVAL ON THE FULL TIER
	for thisInterval from 1 to numberOfPhonemes

    	# GET THE LABEL OF THE INTERVAL
    	select TextGrid 'thisTextGrid$'
    	thisPhoneme$ = Get label of interval: 4, thisInterval
		# (3 if by rep, 4 if by full)
		
		if thisPhoneme$ != ""
    
    		# CALCULATE RATE
			thisPhonemeStartTime = Get start point: 4, thisInterval
			# (3 if by rep, 4 if by full)
    		thisPhonemeEndTime = Get end point: 4, thisInterval
			# (3 if by rep, 4 if by full)
    		duration = thisPhonemeEndTime - thisPhonemeStartTime
			ddkrate = 9/duration
			# (3 if by rep, 9 if by full)
			outputLine$ = filename$ + "," + thisPhoneme$ + "," + string$ (duration) + "," + string$ (ddkrate)
			appendFileLine: "/Users/hannahrowe/Desktop/Duration_Data_For_R.csv", outputLine$
		endif
	endfor
endfor