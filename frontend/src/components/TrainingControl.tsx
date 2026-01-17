import React from 'react';
import { Card, CardContent, Typography, Button, Box, Stack, CircularProgress } from '@mui/material';
import CloudDownloadIcon from '@mui/icons-material/CloudDownload';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import FeaturedPlayListIcon from '@mui/icons-material/FeaturedPlayList';

interface TrainingControlProps {
  onDownloadData: () => void;
  onGenerateFeatures: () => void;
  onTrainModel: () => void;
  isDownloading: boolean;
  isGenerating: boolean;
  isTraining: boolean;
}

const TrainingControl: React.FC<TrainingControlProps> = ({
  onDownloadData,
  onGenerateFeatures,
  onTrainModel,
  isDownloading,
  isGenerating,
  isTraining,
}) => {
  return (
    <Card sx={{ minWidth: 275, p: 2 }}>
      <CardContent>
        <Typography variant="h5" component="div" gutterBottom>
          Pipeline Control
        </Typography>
        <Stack spacing={2}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="body1">1. Data Collection</Typography>
            <Button
              variant="contained"
              startIcon={isDownloading ? <CircularProgress size={20} color="inherit" /> : <CloudDownloadIcon />}
              onClick={onDownloadData}
              disabled={isDownloading || isGenerating || isTraining}
            >
              {isDownloading ? 'Downloading...' : 'Download Data'}
            </Button>
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="body1">2. Feature Engineering</Typography>
            <Button
              variant="contained"
              color="secondary"
              startIcon={isGenerating ? <CircularProgress size={20} color="inherit" /> : <FeaturedPlayListIcon />}
              onClick={onGenerateFeatures}
              disabled={isDownloading || isGenerating || isTraining}
            >
              {isGenerating ? 'Generating...' : 'Generate Features'}
            </Button>
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="body1">3. Model Training</Typography>
            <Button
              variant="contained"
              color="success"
              startIcon={isTraining ? <CircularProgress size={20} color="inherit" /> : <ModelTrainingIcon />}
              onClick={onTrainModel}
              disabled={isDownloading || isGenerating || isTraining}
            >
              {isTraining ? 'Training...' : 'Train LightGBM'}
            </Button>
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
};

export default TrainingControl;
