import { useState, useEffect } from 'react'
import axios from 'axios'
import { Container, Grid, Typography, Box, AppBar, Toolbar, Alert, Snackbar } from '@mui/material'
import ShowChartIcon from '@mui/icons-material/ShowChart'
import PredictionCard from './components/PredictionCard'
import TrainingControl from './components/TrainingControl'
import MetricsDashboard from './components/MetricsDashboard'

// Configure Axios
const API_URL = 'http://localhost:8000'; // Adjust if needed

function App() {
  const [prediction, setPrediction] = useState<string | null>(null);
  const [probability, setProbability] = useState<number | null>(null);
  const [metrics, setMetrics] = useState<any>(null);

  const [isDownloading, setIsDownloading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);

  const [message, setMessage] = useState<{text: string, type: 'success' | 'error' | 'info'} | null>(null);

  useEffect(() => {
    // Load initial metrics if available
    fetchMetrics();
    // Load initial prediction
    fetchPrediction();
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API_URL}/metrics`);
      setMetrics(response.data);
    } catch (error) {
      console.log('Metrics not available yet');
    }
  };

  const fetchPrediction = async () => {
    setIsPredicting(true);
    try {
      const response = await axios.get(`${API_URL}/predict`);
      setPrediction(response.data.prediction);
      setProbability(response.data.probability);
    } catch (error) {
      console.log('Prediction not available yet');
      setPrediction(null);
      setProbability(null);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleDownloadData = async () => {
    setIsDownloading(true);
    try {
      await axios.post(`${API_URL}/data/refresh`);
      setMessage({ text: 'Data downloaded successfully!', type: 'success' });
    } catch (error) {
      setMessage({ text: 'Failed to download data.', type: 'error' });
    } finally {
      setIsDownloading(false);
    }
  };

  const handleGenerateFeatures = async () => {
    setIsGenerating(true);
    try {
      const response = await axios.post(`${API_URL}/features/generate`);
      setMessage({ text: `Features generated: ${response.data.features_count} features created.`, type: 'success' });
    } catch (error) {
      setMessage({ text: 'Failed to generate features.', type: 'error' });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleTrainModel = async () => {
    setIsTraining(true);
    try {
      // Trigger training
      await axios.post(`${API_URL}/train/lightgbm`);
      setMessage({ text: 'Model trained successfully!', type: 'success' });

      // Refresh metrics and prediction
      await fetchMetrics();
      await fetchPrediction();
    } catch (error) {
      setMessage({ text: 'Failed to train model.', type: 'error' });
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <Box sx={{ flexGrow: 1, bgcolor: '#f5f5f5', minHeight: '100vh' }}>
      <AppBar position="static" color="primary">
        <Toolbar>
          <ShowChartIcon sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            SPY Price Prediction AI
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Grid container spacing={3}>
          {/* Top Row: Prediction & Controls */}
          <Grid size={{ xs: 12, md: 6 }}>
            <PredictionCard
              prediction={prediction}
              probability={probability}
              isLoading={isPredicting}
              error={null}
            />
          </Grid>
          <Grid size={{ xs: 12, md: 6 }}>
            <TrainingControl
              onDownloadData={handleDownloadData}
              onGenerateFeatures={handleGenerateFeatures}
              onTrainModel={handleTrainModel}
              isDownloading={isDownloading}
              isGenerating={isGenerating}
              isTraining={isTraining}
            />
          </Grid>

          {/* Bottom Row: Metrics & Details */}
          <Grid size={12}>
            <MetricsDashboard metrics={metrics} />
          </Grid>
        </Grid>
      </Container>

      <Snackbar
        open={!!message}
        autoHideDuration={6000}
        onClose={() => setMessage(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={() => setMessage(null)} severity={message?.type || 'info'} sx={{ width: '100%' }}>
          {message?.text}
        </Alert>
      </Snackbar>
    </Box>
  )
}

export default App
