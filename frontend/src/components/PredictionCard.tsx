import React from 'react';
import { Card, CardContent, Typography, Box, CircularProgress } from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

interface PredictionCardProps {
  prediction: string | null;
  probability: number | null;
  isLoading: boolean;
  error: string | null;
}

const PredictionCard: React.FC<PredictionCardProps> = ({ prediction, probability, isLoading, error }) => {
  if (isLoading) {
    return (
      <Card sx={{ minWidth: 275, textAlign: 'center', p: 2 }}>
        <CardContent>
          <CircularProgress />
          <Typography sx={{ mt: 2 }} color="text.secondary">
            Loading prediction...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card sx={{ minWidth: 275, textAlign: 'center', p: 2, bgcolor: '#ffebee' }}>
        <CardContent>
          <Typography color="error">
            {error}
          </Typography>
        </CardContent>
      </Card>
    );
  }

  if (!prediction || probability === null) {
    return (
      <Card sx={{ minWidth: 275, textAlign: 'center', p: 2 }}>
        <CardContent>
          <Typography color="text.secondary">
            No prediction available. Train the model and click Predict.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const isUp = prediction === 'UP';
  const confidence = (probability * 100).toFixed(2);

  return (
    <Card sx={{ minWidth: 275, textAlign: 'center', p: 2, bgcolor: isUp ? '#e8f5e9' : '#ffebee' }}>
      <CardContent>
        <Typography sx={{ fontSize: 14 }} color="text.secondary" gutterBottom>
          3-Day Forecast
        </Typography>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', mb: 1 }}>
          {isUp ? (
            <TrendingUpIcon sx={{ fontSize: 60, color: 'green' }} />
          ) : (
            <TrendingDownIcon sx={{ fontSize: 60, color: 'red' }} />
          )}
        </Box>
        <Typography variant="h3" component="div" color={isUp ? 'green' : 'red'} fontWeight="bold">
          {prediction}
        </Typography>
        <Typography sx={{ mb: 1.5 }} color="text.secondary">
          Confidence: {confidence}%
        </Typography>
        <Typography variant="body2">
          Model: LightGBM (Selected Features)
        </Typography>
      </CardContent>
    </Card>
  );
};

export default PredictionCard;
