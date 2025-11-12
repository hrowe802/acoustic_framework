formantData = readtable('/Users/hannahrowe/Google Drive/My Drive/Research/Scripts/2 Matlab Scripts/Formant_Data_For_Matlab.csv', 'Format', '%s%s%f%f%f%f')

% Display numbers in normal\ notation instead of scientific notation
format long g

% Make an empty table to store results
resultsTable = cell2table(cell(0,17), 'VariableNames', {'Participant', 'Task', 'Vow', 'OnsetFreq', 'OffsetFreq', 'Range', 'Slope', 'F1xF2Xcorr', 'F1xF2Corr', 'F1xF2Cov', 'Ratio', 'F1Vel', 'F1Accel', 'F1Jerk', 'F2Vel', 'F2Accel', 'F2Jerk'})

% Separate out each participant
uniqueParticipants = unique(formantData.Participant);
for participantIndex = 1:length(uniqueParticipants)
    thisParticipant = uniqueParticipants(participantIndex);
    participantStringRows = string(formantData.Participant);
    participantRows = participantStringRows == thisParticipant;
    participantData = formantData(participantRows, :);

    % Separate out each task within each participant
    uniqueTasks = unique(participantData.Task);
    for taskIndex = 1:length(uniqueTasks)
        thisTask = uniqueTasks(taskIndex);
        taskStringRows = string(participantData.Task);
        taskRows = taskStringRows == thisTask;
        taskData = participantData(taskRows, :);
        taskNumbers = taskData{:,3:6}; % variable that all calculations are run on (unique task for unique participant)

        % Define sampling frequency
        Fs = 0.05;

        % Process whole duration of each VOT
        if contains(thisTask, 'VOT')
            duration = taskNumbers(1,4);
            thisTable = table(thisParticipant, thisTask, duration, [0, 0], [0, 0], [0, 0], [0, 0], 0, 0, 0, [0, 0], 0, 0, 0, 0, 0, 0, 'VariableNames', {'Participant', 'Task', 'Vow', 'OnsetFreq', 'OffsetFreq', 'Range', 'Slope', 'F1xF2Xcorr', 'F1xF2Corr', 'F1xF2Cov', 'Ratio', 'F1Vel', 'F1Accel', 'F1Jerk', 'F2Vel', 'F2Accel', 'F2Jerk'});
            resultsTable = [resultsTable; thisTable]
            continue
        end

        % Calculate velocity, acceleration, and jerk of first formant
        F1position = taskNumbers(:,2);
        F1time = taskNumbers(:,1);
        F1vel = diff(F1position)./diff(F1time);
        F1vel(end+1,:) = "NA";
        F1accel = diff(F1vel)./diff(F1time);
        F1accel(end+1,:) = "NA";
        F1jerk = diff(F1accel)./diff(F1time);
        F1jerk(end+1,:) = "NA";

        % Calculate velocity, acceleration, and jerk of second formant
        F2position = taskNumbers(:,3);
        F2time = taskNumbers(:,1);
        F2vel = diff(F2position)./diff(F2time);
        F2vel(end+1,:) = "NA";
        F2accel = diff(F2vel)./diff(F2time);
        F2accel(end+1,:) = "NA";
        F2jerk = diff(F2accel)./diff(F2time);
        F2jerk(end+1,:) = "NA";

        % Append velocity, acceleration, and jerk to taskNumbers
        taskNumbers = [taskNumbers, F1vel, F1accel, F1jerk, F2vel, F2accel, F2jerk];

        % Our final results show full duration of all tasks
        duration = taskNumbers(1,4);

        % Our final results show mean velocity, acceleration, and jerk of all tasks
        F1vel = nanmean(taskNumbers(:,5));
        F1accel = nanmean(taskNumbers(:,6));
        F1jerk = nanmean(taskNumbers(:,7));
        F2vel = nanmean(taskNumbers(:,8));
        F2accel = nanmean(taskNumbers(:,9));
        F2jerk = nanmean(taskNumbers(:,10));

        % Only process the first half of each puh, tuh, and kuh to avoid coarticulatory effects
        taskNumbersSize = size(taskNumbers);
        taskNumbersRowCount = taskNumbersSize(1);
        midPoint = round(taskNumbersRowCount/2);

        % Split out into time vector and formant matrix
        timeMat = taskNumbers(1:midPoint,1);
        dataHalfMat = taskNumbers(1:midPoint,2:3);

        % Calculate slopes of F1 and F2 (using half duration)
        dur = timeMat(end) - timeMat(1);
        onsetFreq = [dataHalfMat(1,1); dataHalfMat(1,2)];
        offsetFreq = [dataHalfMat(end,1); dataHalfMat(end,2)];
        rangeMat = offsetFreq-onsetFreq;
        slopeMat = (offsetFreq-onsetFreq)/dur;

        % Get cross correlation, correlation, and covariance matrices of F1xF2
        dataXcorr = xcorr(taskNumbers(:,2:3), 'coeff');
        dataCorr = corr(taskNumbers(:,2:3));
        dataCov = cov(taskNumbers(:,2:3));

        % Get ratios of F1/F2 onset freqs to F1/F2 offset freqs
        ratio = onsetFreq(1:2)./offsetFreq(1:2);

        % Create table of all data
        thisTable = table(thisParticipant, thisTask, duration, onsetFreq', offsetFreq', rangeMat', slopeMat'/1000, dataXcorr(2,1), dataCorr(2,1), dataCov(2,1), ratio', F1vel, F1accel, F1jerk, F2vel, F2accel, F2jerk, 'VariableNames', {'Participant', 'Task', 'Vow', 'OnsetFreq', 'OffsetFreq', 'Range', 'Slope', 'F1xF2Xcorr', 'F1xF2Corr', 'F1xF2Cov', 'Ratio', 'F1Vel', 'F1Accel', 'F1Jerk', 'F2Vel', 'F2Accel', 'F2Jerk'});
        resultsTable = [resultsTable; thisTable]
    end
end

writetable(resultsTable, '/Users/hannahrowe/Desktop/Formant_Data_For_Python.csv')
