import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  Container, Grid, Typography, Box, AppBar, Toolbar,
  Alert, Snackbar, ThemeProvider, createTheme, CssBaseline, Stack
} from '@mui/material'
import ShowChartIcon from '@mui/icons-material/ShowChart'
import PredictionCard from './components/PredictionCard'
import TrainingControl from './components/TrainingControl'
import MetricsDashboard from './components/MetricsDashboard'

// Dark Mode Theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#90caf9' },
    secondary: { main: '#ce93d8' },
    background: { default: '#121212', paper: '#1e1e1e' },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});

const API_URL = 'http://localhost:8000';

function App() {
  const [selectedModel, setSelectedModel] = useState<string>('lightgbm');

  const [prediction, setPrediction] = useState<string | null>(null);
  const [probability, setProbability] = useState<number | null>(null);
  const [metrics, setMetrics] = useState<any>(null);

  const [isDownloading, setIsDownloading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);

  const [message, setMessage] = useState<{text: string, type: 'success' | 'error' | 'info'} | null>(null);

  // Fetch data when model selection changes
  useEffect(() => {
    fetchMetrics();
    fetchPrediction();
  }, [selectedModel]);

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API_URL}/metrics?model=${selectedModel}`);
      setMetrics(response.data);
    } catch (error) {
      setMetrics(null); // Reset if model not trained yet
    }
  };

  const fetchPrediction = async () => {
    setIsPredicting(true);
    try {
      const response = await axios.get(`${API_URL}/predict?model=${selectedModel}`);
      setPrediction(response.data.prediction);
      setProbability(response.data.probability);
    } catch (error) {
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
      setMessage({ text: `Features generated: ${response.data.features_count} created.`, type: 'success' });
    } catch (error) {
      setMessage({ text: 'Failed to generate features.', type: 'error' });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleTrainModel = async () => {
    setIsTraining(true);
    try {
      setMessage({ text: `Training ${selectedModel}... This may take a moment.`, type: 'info' });
      await axios.post(`${API_URL}/train/${selectedModel}`);
      setMessage({ text: `${selectedModel} trained successfully!`, type: 'success' });
      await fetchMetrics();
      await fetchPrediction();
    } catch (error) {
      setMessage({ text: 'Training failed. Check console for details.', type: 'error' });
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1, minHeight: '100vh' }}>
        <AppBar position="static" color="transparent" enableColorOnDark elevation={0} sx={{ borderBottom: '1px solid #333' }}>
          <Toolbar>
            <ShowChartIcon sx={{ mr: 2, color: 'primary.main' }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
              SPY Price Prediction AI
            </Typography>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
          <Grid container spacing={3}>
            {/* Left Col: Controls & Prediction */}
            <Grid size={{ xs: 12, md: 4 }}>
              <Stack spacing={3}>
                <TrainingControl
                  selectedModel={selectedModel}
                  onModelChange={setSelectedModel}
                  onDownloadData={handleDownloadData}
                  onGenerateFeatures={handleGenerateFeatures}
                  onTrainModel={handleTrainModel}
                  isDownloading={isDownloading}
                  isGenerating={isGenerating}
                  isTraining={isTraining}
                />
                <PredictionCard
                  prediction={prediction}
                  probability={probability}
                  isLoading={isPredicting}
                  modelName={selectedModel}
                />
              </Stack>
            </Grid>

            {/* Right Col: Dashboard */}
            <Grid size={{ xs: 12, md: 8 }}>
              <MetricsDashboard metrics={metrics} modelName={selectedModel} />
            </Grid>
          </Grid>
        </Container>

        <Snackbar
          open={!!message}
          autoHideDuration={6000}
          onClose={() => setMessage(null)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert onClose={() => setMessage(null)} severity={message?.type || 'info'} variant="filled">
            {message?.text}
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  )
}

export default App
