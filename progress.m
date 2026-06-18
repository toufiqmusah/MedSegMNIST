%% MedSegMNIST — Dataset Progress Tracker
% Status legend:  ✅ done  🔜 planned  ⏳ in progress  ❌ blocked

fprintf('╔══════════════════════════════════════════════════════════╗\n');
fprintf('║              MedSegMNIST — Dataset Status               ║\n');
fprintf('╠══════════════════════════════════════════════════════════╣\n');

datasets = {
  1, 'AbdomenSegMNIST3D',  'CT',         '3D',  6,  [64, 96, 128, 192, "native"],  '✅';
  2, 'BrainSegMNIST3D',    'MRI',        '3D',  4,  [96, 128, 224, "native"],       '✅';
  3, 'SpineSegMNIST3D',    'MR',         '3D',  3,  [64, 96, 128, 192, "native"],   '✅';
  4, 'KneeSegMNIST3D',     'MR',         '3D',  6,  [64, 96, 128, 192, "native"],   '✅';
  5, 'LungSegMNIST2D',     'X-ray',      '2D',  2,  [128, 256, 512],                '✅';
  6, 'NucleiSegMNIST2D',   'Pathology',  '2D',  2,  [256, 512, "native"],           '✅';
  7, 'BreastSegMNIST',     'Ultrasound', '2D',  2,  [128, 256, "native"],           '✅';
  8, 'FundusSegMNIST2D',   'Fundus',     '2D',  2,  [256, 512, 1024, "native"],     '✅';
  9, 'SkinSegMNIST2D',     'Dermoscopy', '2D',  2,  [128, 256, 512, "native"],      '✅';
 10, 'PolypSegMNIST2D',    'Endoscopy',  '2D',  2,  [128, 256, 512, "native"],      '✅';
};

for i = 1:size(datasets, 1)
    fprintf('║ %2d  %-19s  %-12s  %-2s  %2d classes  ✅\n', ...
        datasets{i,:});
end

fprintf('╚══════════════════════════════════════════════════════════╝\n');
fprintf('\nTotal datasets: %d\n', size(datasets, 1));
fprintf('Status: All 10 datasets have been registered and are ready for preprocessing.\n');
