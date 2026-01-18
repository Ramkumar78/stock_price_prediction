import React from 'react';
import {
  Card, CardContent, Typography, Paper, Grid, Box, Chip
} from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

interface MetricsDashboardProps {
  metrics: any;
  modelName: string;
}

const MetricsDashboard: React.FC<MetricsDashboardProps> = ({ metrics, modelName }) => {
  if (!metrics) {
    return (
      <Card sx={{ height: '100%', borderRadius: 4, display: 'flex', alignItems: 'center', justifyContent: 'center', p: 4, bgcolor: 'background.paper' }}>
        <Typography color="text.secondary">
          No metrics found for {modelName}. Please train the model.
        </Typography>
      </Card>
    );
  }

  const { train, test } = metrics.metrics;

  // Prepare data for chart
  const chartData = [
    { name: 'ROC AUC', Train: train.roc_auc, Test: test.roc_auc },
    { name: 'Accuracy', Train: train.accuracy, Test: test.accuracy },
    { name: 'Precision', Train: train.precision, Test: test.precision },
    { name: 'Recall', Train: train.recall, Test: test.recall },
  ];

  return (
    <Card sx={{ height: '100%', borderRadius: 4, boxShadow: 3 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
            Performance Analysis
          </Typography>
          <Chip label={modelName.toUpperCase()} color="primary" variant="outlined" />
        </Box>

        <Grid container spacing={3}>
          {/* Key Metric Highlight */}
          <Grid size={{ xs: 12, md: 3 }}>
             <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'rgba(144, 202, 249, 0.1)', mb: 2 }}>
                <Typography variant="subtitle2" color="text.secondary">Test Accuracy</Typography>
                <Typography variant="h4" color="primary" fontWeight="bold">
                  {(test.accuracy * 100).toFixed(1)}%
                </Typography>
             </Paper>
             <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'rgba(206, 147, 216, 0.1)' }}>
                <Typography variant="subtitle2" color="text.secondary">Test ROC AUC</Typography>
                <Typography variant="h4" color="secondary" fontWeight="bold">
                  {test.roc_auc.toFixed(3)}
                </Typography>
             </Paper>
          </Grid>

          {/* Chart */}
          <Grid size={{ xs: 12, md: 9 }}>
            <Box sx={{ height: 300, width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#333', border: 'none' }}
                    itemStyle={{ color: '#fff' }}
                    formatter={(value: any) => typeof value === 'number' ? value.toFixed(3) : value}
                  />
                  <Legend />
                  <Bar dataKey="Train" fill="#90caf9" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Test" fill="#ce93d8" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default MetricsDashboard;
