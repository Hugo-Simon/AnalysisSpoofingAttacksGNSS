%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright 2015-2021 Finnish Geospatial Research Institute FGI, National
%% Land Survey of Finland. This file is part of FGI-GSRx software-defined
%% receiver. FGI-GSRx is a free software: you can redistribute it and/or
%% modify it under the terms of the GNU General Public License as published
%% by the Free Software Foundation, either version 3 of the License, or any
%% later version. FGI-GSRx software receiver is distributed in the hope
%% that it will be useful, but WITHOUT ANY WARRANTY, without even the
%% implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
%% See the GNU General Public License for more details. You should have
%% received a copy of the GNU General Public License along with FGI-GSRx
%% software-defined receiver. If not, please visit the following website 
%% for further information: https://www.gnu.org/licenses/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [trackResults]= doTracking(acqResults, allSettings)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function takes input of acquisition results and performs tracking.
%
% Inputs:
%   acqResults      - Results from signal acquisition for all signals
%   allSettings     - Receiver settings
%
% Outputs:
%   trackResults    - Results from signal tracking for all signals
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start timer for tracking
trackStartTime = now;

% UI output
disp (['   Tracking started at ', datestr(trackStartTime)]); 

% Initialise tracking structure
trackResults = initTracking(acqResults, allSettings);  

% Flag: si true imprimimos en consola las variables; si false solo guardamos en CSV
printConsole = false;

% Eliminar ficheros tracking_*.csv al iniciar (evita hacerlo dentro del bucle)
oldCSVs = dir('tracking_*.csv');
for k = 1:numel(oldCSVs)
    try
        delete(oldCSVs(k).name);
    catch
        warning('doTracking:DeleteFailed', 'Could not delete %s', oldCSVs(k).name);
    end
end

% Let's loop over all enabled signals and open files for reading
for signalNr = 1:allSettings.sys.nrOfSignals

    % Extract block of parameters for one signal from settings
    signal = allSettings.sys.enabledSignals{signalNr};
    signalSettings = allSettings.(signal);
    
    % Open file for reading
    [fid, message] = fopen(signalSettings.rfFileName, 'rb');
    if (fid == -1)
       error('Failed to open data file for tracking!');
       return;
    else
       fidTemp{signalNr} = fid;
       % Inicializar estructuras para ficheros CSV por canal (se crearán luego)
       % csvFidChannel{signalNr,channelNr} almacenará los file ids por canal
       % csvRowWritten{signalNr,channelNr} indicará si ya se escribió la primera fila
       % (se inicializan a vacío; se crean cuando se escribe la primera vez)
       % No abrir ficheros CSV a nivel de señal aquí porque queremos uno por PRN/canal
       
    end
end

t1=clock;

for loopCnt =  1:allSettings.sys.msToProcess % Loop over all epochs
    for signalNr = 1:allSettings.sys.nrOfSignals % Loop over all signals
        signal = allSettings.sys.enabledSignals{signalNr};
        trackResults.(signal).loopCnt = loopCnt;
        for channelNr = 1:trackResults.(signal).nrObs % Loop over all channels            
             % Set file pointer
            trackResults.(signal).fid = fidTemp{signalNr};

            % Check epoch boundary
            if(mod(loopCnt,allSettings.(signal).codeLengthMs)==0)
                
                % Correlate signal
                trackResults.(signal) = GNSSCorrelation(allSettings.(signal),trackResults.(signal),channelNr);             
                
                % Tracking of signal
                trackResults.(signal) = GNSSTracking(allSettings.(signal),trackResults.(signal),channelNr); 

                prn = trackResults.(signal).channel(channelNr).SvId.satId;
                
                % Mostrar siguiente elemento secuencial de varias variables por canal
                try
                    chref = trackResults.(signal).channel(channelNr);
                    vars = {'CN0fromSNR','meanCN0fromSNR','I_P','Q_P','doppler', 'pllLockIndicator', 'dllDiscr', 'carrFreq'}; % variables a mostrar
                    outVals = cell(1,numel(vars));
                    for vi = 1:numel(vars)
                        varname = vars{vi};
                        if isfield(chref, varname)
                            v = chref.(varname);
                            idx_field = [varname '_print_idx'];
                            if isfield(chref, idx_field) && ~isempty(chref.(idx_field))
                                idx = chref.(idx_field);
                            else
                                idx = 1;
                            end
                            n = numel(v);
                            if n >= idx && idx >= 1
                                try
                                    % extraer el elemento robustamente (soporta cell que contengan numéricos)
                                    if iscell(v)
                                        try
                                            raw = v{idx};
                                        catch
                                            raw = v(idx);
                                        end
                                    else
                                        raw = v(idx);
                                    end
                                    if isnumeric(raw) && isscalar(raw)
                                        val = double(raw);
                                        outVals{vi} = val;
                                    elseif ischar(raw)
                                        s = raw;                                        
                                        outVals{vi} = s;
                                    else
                                        try
                                            s = mat2str(raw);
                                        catch
                                            s = sprintf('<%s>', class(raw));
                                        end
                                        outVals{vi} = s;
                                    end
                                catch
                                    outVals{vi} = '';
                                end
                                % Guardar índice incrementado en la estructura principal
                                trackResults.(signal).channel(channelNr).(idx_field) = idx + 1;
                            else
                                outVals{vi} = '';
                            end
                        else
                            outVals{vi} = '';
                        end
                    end
                    % Volcar la fila al CSV si se abrió el fichero
                    try
                        % Crear/usar fichero CSV por canal (concatenando PRN)
                        if exist('prn','var') && ~isempty(prn) && ~isnan(double(prn))
                            chIdx = channelNr;
                            % Comprobar si ya tenemos un fid para este canal
                            needOpen = true;
                            if exist('csvFidChannel','var') && size(csvFidChannel,1) >= signalNr && size(csvFidChannel,2) >= chIdx
                                if ~isempty(csvFidChannel{signalNr,chIdx}) && csvFidChannel{signalNr,chIdx} ~= -1
                                    fidcsv = csvFidChannel{signalNr,chIdx};
                                    needOpen = false;
                                end
                            end
                            if needOpen
                                csvFileNameCh = sprintf('tracking_%s_%d.csv', signal, double(prn));
                                % eliminar fichero previo sólo la primera vez por canal
                                if exist(csvFileNameCh,'file') == 2
                                    % no eliminar para no perder datos si reanuda; dejamos en append
                                end
                                fidcsv = fopen(csvFileNameCh,'a');
                                if fidcsv == -1
                                    % marcar como no disponible
                                    csvFidChannel{signalNr,chIdx} = -1;
                                else
                                    csvFidChannel{signalNr,chIdx} = fidcsv;
                                    if ftell(fidcsv) == 0
                                        fprintf(fidcsv, 'svid,CN0fromSNR,meanCN0fromSNR,I_P,Q_P,doppler,pllLockIndicator,dllDiscr,carrFreq\n'); % variables a mostrar
                                    end
                                    csvRowWritten{signalNr,chIdx} = false;
                                end
                            end

                            % Si el fichero está disponible, escribir (saltando la primera fila por canal)
                            if exist('csvFidChannel','var') && size(csvFidChannel,1) >= signalNr && size(csvFidChannel,2) >= chIdx && csvFidChannel{signalNr,chIdx} ~= -1
                                fidcsv = csvFidChannel{signalNr,chIdx};
                                if exist('csvRowWritten','var') && size(csvRowWritten,1) >= signalNr && size(csvRowWritten,2) >= chIdx && ~csvRowWritten{signalNr,chIdx}
                                    % marcar y no escribir la primera fila
                                    csvRowWritten{signalNr,chIdx} = true;
                                else
                                    % escribir prn (si disponible)
                                    fprintf(fidcsv, '%d,', double(prn));
                                    for j = 1:numel(outVals)
                                        vj = outVals{j};
                                        if isnumeric(vj) && ~isempty(vj)
                                            if j == numel(outVals)
                                                fprintf(fidcsv, '%g\n', double(vj));
                                            else
                                                fprintf(fidcsv, '%g,', double(vj));
                                            end
                                        else
                                            if isempty(vj)
                                                if j == numel(outVals)
                                                    fprintf(fidcsv, '\n');
                                                else
                                                    fprintf(fidcsv, ',');
                                                end
                                            else
                                                sj = strrep(vj, '"', '""');
                                                if j == numel(outVals)
                                                    fprintf(fidcsv, '"%s"\n', sj);
                                                else
                                                    fprintf(fidcsv, '"%s",', sj);
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    catch
                        % evitar errores de escritura afecten al tracking
                    end
                catch
                    % evitar que fallen las impresiones afecten al tracking
                end
            end
        end
    end

    % UI function
    if (mod(loopCnt, 1000) == 0)   
        t2 = clock;
        time = etime(t2,t1);
        estimtime = allSettings.sys.msToProcess/loopCnt * time;
        showTrackStatus(trackResults,allSettings,loopCnt);
        msProcessed = loopCnt;
        msLeftToProcess = allSettings.sys.msToProcess-loopCnt;
        disp(['Ms Processed: ',int2str(msProcessed),' Ms Left: ',int2str(msLeftToProcess)]);
        disp(['Time processed: ',int2str(time),' Time left: ',int2str(estimtime-time)]);

     end    

end % Loop over all epochs

% Notify user tracking is over
disp(['   Tracking is over (elapsed time ', datestr(now - trackStartTime, 13), ')']);
if fidcsv ~= -1
    fullPath = fullfile(pwd, csvFileNameCh);
    disp(['CSV file path: ', fullPath]);
end


