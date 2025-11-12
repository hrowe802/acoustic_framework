project_dir$ = "/Users/hannahrowe/Google Drive/My Drive/Research/Acoustic Files/Analyzed/Text Grids (ALS-HC-PA-PD-PPA) ORIGINAL/"

# SETUP THE CSV FILE HEADER
writeFileLine: "/Users/hannahrowe/Desktop/Prediction_Data_For_Python.csv", "starting script..."

# MAKE A LIST OF ALL TEXTGRIDS IN THE FOLDER
Create Strings as file list... list 'project_dir$'*.TextGrid
file_list = selected("Strings")
file_count = Get number of strings

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

	# EXTRACT THE NAMES OF PRAAT OBJECTS
	thisSound$ = selected$("Sound")
	thisTextGrid$ = selected$("TextGrid")

	# EXTRACT THE NUMBER OF INTERVALS IN THE CONSONANT TIER
	select TextGrid 'thisTextGrid$'
	numberOfPhonemes = Get number of intervals: 2

	# CREATE THE FORMANT OBJECT
	select Sound 'thisSound$'
	To Formant (burg)... 0 5 5000 0.0025 50

	# LOOP THROUGH EACH INTERVAL ON THE CONSONANT TIER
	for thisInterval from 1 to numberOfPhonemes

    	# GET THE LABEL OF THE INTERVAL
    	select TextGrid 'thisTextGrid$'
    	thisPhoneme$ = Get label of interval: 2, thisInterval
		
		if thisPhoneme$ != ""
    
    		# EXTRACT SPECTRAL MOMENTS
			thisPhonemeStartTime = Get start point: 2, thisInterval
    		thisPhonemeEndTime = Get end point: 2, thisInterval
			select Sound 'filename$'
			call spectralCorrelation 'thisPhonemeStartTime' 'thisPhonemeEndTime'
		endif
	endfor
endfor

procedure spectralCorrelation .onset .offset
	# SET WINDOW LENGTH
	Extract part... '.onset' '.offset' Hamming 1 no
	snd = selected ("Sound")

	# SET PREEMPHASIS FILTER
	Filter (pre-emphasis)... 50
	sndPre = selected ("Sound")
	
	# CALCULATE SPECTRUM
	To LPC (burg)... 16 0.025 0.005 50.0
	To Spectrum (slice)... 0.0 20.0 0.0 50.0

	# EXTRACT SPECTRAL CORRELATION
	myList$ = List: "no", "no", "no", "no", "no", "yes"
	appendFileLine: "/Users/hannahrowe/Desktop/Prediction_Data_For_Python.csv", "new phoneme", "|", thisSound$, "|", thisPhoneme$
	appendFileLine: "/Users/hannahrowe/Desktop/Prediction_Data_For_Python.csv", myList$
endproc