project_dir$ = "/Users/hannahrowe/Google Drive/My Drive/Research/Acoustic Files/Analyzed/Text Grids (***)/"

# MAKE A LIST OF ALL TEXTGRIDS IN THE FOLDER
Create Strings as file list... list 'project_dir$'*.TextGrid
file_list = selected("Strings")
file_count = Get number of strings

# SETUP THE CSV FILE HEADER
writeFileLine: "/Users/hannahrowe/Desktop/Spectrum_Data_For_Python.csv", "Participant, Task, CentralGravity, StandardDeviation, Skewness, Kurtosis"

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
			call spectralMoments 'thisPhonemeStartTime' 'thisPhonemeEndTime'
			outputLine$ = filename$ + "," + thisPhoneme$ + "," + string$ (grav) + "," + string$ (sdev) + "," + string$ (skew) + "," + string$ (kurt)
			appendFileLine: "/Users/hannahrowe/Desktop/Spectrum_Data_For_Python.csv", outputLine$
		endif
	endfor
endfor

procedure spectralMoments .onset .offset
	# SET WINDOW LENGTH
	Extract part... '.onset' '.offset' Hamming 1 no
	snd = selected ("Sound")

	# SET PREEMPHASIS FILTER
	Filter (pre-emphasis)... 100
	sndPre = selected ("Sound")

	# CALCULATE SPECTRUM
	To Spectrum... yes

	# EXTRACT SPECTRAL MOMENTS
	grav = Get centre of gravity... 2
	sdev = Get standard deviation... 2
	skew = Get skewness... 2
	kurt = Get kurtosis... 2
	#printline 'grav:2''tab$''sdev:2''tab$''skew:4''tab$''kurt:4'
	#plus snd
	#plus sndPre
	Remove
endproc