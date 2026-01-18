import { useState, useEffect } from 'react';
import {
  Box, TextField, Button, MenuItem, Select, FormControl, InputLabel,
  Typography, Card, CardContent, LinearProgress, Alert, Stack, Chip, Divider, Grid
} from '@mui/material';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export default function CustomTrain() {
  const [ticker, setTicker] = useState('TSLA');
  const [model, setModel] = useState('xgboost');
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [progress, setProgress] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let interval: any;
    if (jobId && status !== 'completed' && status !== 'failed') {
      interval = setInterval(async () => {
        try {
          const response = await axios.get(`${API_URL}/custom/status/${jobId}`);
          setStatus(response.data.status);
          setProgress(response.data.progress);
          if (response.data.result) {
            setResult(response.data.result);
          }
          if (response.data.error) {
            setError(response.data.error);
          }
        } catch (e) {
          console.error("Polling error", e);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [jobId, status]);

  const handleStart = async () => {
    setJobId(null);
    setStatus('starting');
    setProgress('Initiating...');
    setResult(null);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/custom/train`, { ticker, model });
      setJobId(response.data.job_id);
      setStatus('pending');
    } catch (e) {
      setError("Failed to start training.");
      setStatus('failed');
    }
  };

  const isRunning = status === 'pending' || status === 'running' || status === 'starting';

  return (
    <Box sx={{ p: 2 }}>
      <Grid container spacing={4}>
        <Grid size={{ xs: 12, md: 5 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Custom Asset Training
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Enter any asset ticker (e.g., TSLA, AAPL, MSFT) to download data, generate features, and train a model on the fly.
              </Typography>

              <Stack spacing={3}>
                <TextField
                  label="Asset Ticker"
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  fullWidth
                  disabled={isRunning}
                />

                <FormControl fullWidth disabled={isRunning}>
                  <InputLabel>Model Type</InputLabel>
                  <Select
                    value={model}
                    label="Model Type"
                    onChange={(e) => setModel(e.target.value)}
                  >
                    <MenuItem value="xgboost">XGBoost</MenuItem>
                    <MenuItem value="lightgbm">LightGBM</MenuItem>
                  </Select>
                </FormControl>

                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleStart}
                  disabled={isRunning || !ticker}
                  size="large"
                >
                  {isRunning ? 'Processing...' : 'Start Pipeline'}
                </Button>

                {isRunning && (
                  <Box>
                    <LinearProgress />
                    <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                      {progress}
                    </Typography>
                  </Box>
                )}

                {error && <Alert severity="error">{error}</Alert>}
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid size={{ xs: 12, md: 7 }}>
          {result ? (
            <Card variant="outlined" sx={{ height: '100%' }}>
              <CardContent>
                 <Typography variant="h5" color="primary" gutterBottom>
                    Result for {result.ticker}
                 </Typography>
                 <Divider sx={{ my: 2 }} />

                 <Grid container spacing={2}>
                    <Grid size={{ xs: 6 }}>
                        <Typography variant="overline" display="block">Prediction (3-Day)</Typography>
                        <Chip
                            label={result.prediction}
                            color={result.prediction === 'UP' ? 'success' : 'error'}
                            sx={{ fontSize: '1.2rem', px: 2, py: 3, borderRadius: 2 }}
                        />
                    </Grid>
                    <Grid size={{ xs: 6 }}>
                        <Typography variant="overline" display="block">Probability</Typography>
                        <Typography variant="h3">
                            {(result.probability * 100).toFixed(1)}%
                        </Typography>
                    </Grid>
                 </Grid>

                 <Box sx={{ mt: 4 }}>
                    <Typography variant="h6" gutterBottom>Model Metrics</Typography>
                    <Grid container spacing={2}>
                        <Grid size={{ xs: 6, sm: 3 }}>
                            <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #333' }}>
                                <Typography variant="caption" color="text.secondary">Test Accuracy</Typography>
                                <Typography variant="h6">{result.metrics.test.accuracy.toFixed(3)}</Typography>
                            </Box>
                        </Grid>
                        <Grid size={{ xs: 6, sm: 3 }}>
                             <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #333' }}>
                                <Typography variant="caption" color="text.secondary">Test ROC AUC</Typography>
                                <Typography variant="h6">{result.metrics.test.roc_auc.toFixed(3)}</Typography>
                            </Box>
                        </Grid>
                         <Grid size={{ xs: 6, sm: 3 }}>
                             <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #333' }}>
                                <Typography variant="caption" color="text.secondary">Train ROC AUC</Typography>
                                <Typography variant="h6">{result.metrics.train.roc_auc.toFixed(3)}</Typography>
                            </Box>
                        </Grid>
                         <Grid size={{ xs: 6, sm: 3 }}>
                             <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #333' }}>
                                <Typography variant="caption" color="text.secondary">F1 Score</Typography>
                                <Typography variant="h6">{result.metrics.test.f1.toFixed(3)}</Typography>
                            </Box>
                        </Grid>
                    </Grid>
                 </Box>

                 <Typography variant="caption" sx={{ mt: 2, display: 'block', color: 'text.secondary' }}>
                    Model: {result.model} | Last Date: {result.last_date}
                 </Typography>
              </CardContent>
            </Card>
          ) : (
            <Box
                sx={{
                    height: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: '2px dashed #333',
                    borderRadius: 2,
                    color: 'text.secondary',
                    p: 4
                }}
            >
                <Typography>
                    Select an asset and start the pipeline to see results here.
                </Typography>
            </Box>
          )}
        </Grid>
      </Grid>
    </Box>
  );
}
