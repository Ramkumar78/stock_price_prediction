import React from 'react';
import { Card, CardContent, Typography, Box, CircularProgress, LinearProgress } from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

interface PredictionCardProps {
  prediction: string | null;
  probability: number | null;
  isLoading: boolean;
  modelName: string;
}

const PredictionCard: React.FC<PredictionCardProps> = ({ prediction, probability, isLoading, modelName }) => {
  const isUp = prediction === 'UP';
  const confidence = probability ? (probability * 100) : 0;
  // If prediction is DOWN, probability usually refers to class 1 (UP), so confidence is 1 - prob
  const displayConfidence = isUp ? confidence : (100 - confidence);

  return (
    <Card sx={{
      borderRadius: 4,
      boxShadow: 3,
      bgcolor: isLoading ? 'background.paper' : (isUp ? 'rgba(27, 94, 32, 0.1)' : 'rgba(183, 28, 28, 0.1)'),
      border: isLoading ? '1px solid transparent' : (isUp ? '1px solid #4caf50' : '1px solid #ef5350'),
      transition: 'all 0.3s ease'
    }}>
      <CardContent sx={{ textAlign: 'center', py: 4 }}>
        <Typography variant="overline" color="text.secondary">
          3-Day Forecast ({modelName})
        </Typography>

        {isLoading ? (
          <Box sx={{ my: 4 }}>
            <CircularProgress color="inherit" />
            <Typography variant="body2" sx={{ mt: 2 }}>Analyzing Market Data...</Typography>
          </Box>
        ) : !prediction ? (
          <Box sx={{ my: 4, opacity: 0.5 }}>
            <Typography variant="h5">No Signal</Typography>
            <Typography variant="body2">Waiting for prediction</Typography>
          </Box>
        ) : (
          <>
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', my: 2 }}>
              {isUp ? (
                <TrendingUpIcon sx={{ fontSize: 80, color: '#66bb6a' }} />
              ) : (
                <TrendingDownIcon sx={{ fontSize: 80, color: '#ef5350' }} />
              )}
            </Box>

            <Typography variant="h2" component="div" sx={{ fontWeight: 'bold', color: isUp ? '#66bb6a' : '#ef5350' }}>
              {prediction}
            </Typography>

            <Box sx={{ mt: 3, mx: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">Confidence</Typography>
                <Typography variant="body2" fontWeight="bold">{displayConfidence.toFixed(1)}%</Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={displayConfidence}
                color={isUp ? "success" : "error"}
                sx={{ height: 10, borderRadius: 5 }}
              />
            </Box>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default PredictionCard;
