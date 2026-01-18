import React from 'react';
import {
  Card, CardContent, Typography, Button, Box, Stack, CircularProgress,
  FormControl, InputLabel, Select, MenuItem, Divider
} from '@mui/material';
import CloudDownloadIcon from '@mui/icons-material/CloudDownload';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import FeaturedPlayListIcon from '@mui/icons-material/FeaturedPlayList';

interface TrainingControlProps {
  selectedModel: string;
  onModelChange: (model: string) => void;
  onDownloadData: () => void;
  onGenerateFeatures: () => void;
  onTrainModel: () => void;
  isDownloading: boolean;
  isGenerating: boolean;
  isTraining: boolean;
}

const TrainingControl: React.FC<TrainingControlProps> = ({
  selectedModel,
  onModelChange,
  onDownloadData,
  onGenerateFeatures,
  onTrainModel,
  isDownloading,
  isGenerating,
  isTraining,
}) => {
  return (
    <Card sx={{ borderRadius: 4, boxShadow: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', mb: 3 }}>
          Control Panel
        </Typography>

        <Stack spacing={3}>
          {/* Step 1: Data */}
          <Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="subtitle2" color="text.secondary">Step 1: Data</Typography>
              {isDownloading && <CircularProgress size={16} />}
            </Box>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<CloudDownloadIcon />}
              onClick={onDownloadData}
              disabled={isDownloading || isGenerating || isTraining}
            >
              Refresh Market Data
            </Button>
          </Box>

          {/* Step 2: Features */}
          <Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="subtitle2" color="text.secondary">Step 2: Engineering</Typography>
              {isGenerating && <CircularProgress size={16} />}
            </Box>
            <Button
              fullWidth
              variant="outlined"
              color="secondary"
              startIcon={<FeaturedPlayListIcon />}
              onClick={onGenerateFeatures}
              disabled={isDownloading || isGenerating || isTraining}
            >
              Generate 115 Features
            </Button>
          </Box>

          <Divider />

          {/* Step 3: Model */}
          <Box>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>Step 3: AI Training</Typography>
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel>Model Architecture</InputLabel>
              <Select
                value={selectedModel}
                label="Model Architecture"
                onChange={(e) => onModelChange(e.target.value)}
                disabled={isTraining}
              >
                <MenuItem value="lightgbm">LightGBM (Recommended)</MenuItem>
                <MenuItem value="xgboost">XGBoost</MenuItem>
                <MenuItem value="catboost">CatBoost</MenuItem>
                <MenuItem value="ensemble">Ensemble (Voting)</MenuItem>
              </Select>
            </FormControl>

            <Button
              fullWidth
              variant="contained"
              color="primary"
              size="large"
              startIcon={isTraining ? <CircularProgress size={20} color="inherit" /> : <ModelTrainingIcon />}
              onClick={onTrainModel}
              disabled={isDownloading || isGenerating || isTraining}
              sx={{ boxShadow: 2 }}
            >
              {isTraining ? 'Training Model...' : `Train ${selectedModel}`}
            </Button>
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
};

export default TrainingControl;
