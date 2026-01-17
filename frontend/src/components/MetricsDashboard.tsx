import React from 'react';
import { Card, CardContent, Typography, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';

interface MetricsDashboardProps {
  metrics: any;
}

const MetricsDashboard: React.FC<MetricsDashboardProps> = ({ metrics }) => {
  if (!metrics) {
    return (
      <Card sx={{ minWidth: 275, p: 2 }}>
        <CardContent>
          <Typography variant="h5" component="div" gutterBottom>
            Model Performance
          </Typography>
          <Typography color="text.secondary">
            No metrics available. Train the model to see results.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const { train, test } = metrics.metrics;

  return (
    <Card sx={{ minWidth: 275, p: 2 }}>
      <CardContent>
        <Typography variant="h5" component="div" gutterBottom>
          Model Performance (LightGBM)
        </Typography>

        <TableContainer component={Paper} variant="outlined">
          <Table aria-label="metrics table">
            <TableHead>
              <TableRow>
                <TableCell>Metric</TableCell>
                <TableCell align="right">Train Set</TableCell>
                <TableCell align="right">Test Set</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              <TableRow>
                <TableCell component="th" scope="row">ROC AUC</TableCell>
                <TableCell align="right">{(train.roc_auc * 100).toFixed(2)}%</TableCell>
                <TableCell align="right" sx={{ fontWeight: 'bold' }}>{(test.roc_auc * 100).toFixed(2)}%</TableCell>
              </TableRow>
              <TableRow>
                <TableCell component="th" scope="row">Accuracy</TableCell>
                <TableCell align="right">{(train.accuracy * 100).toFixed(2)}%</TableCell>
                <TableCell align="right">{(test.accuracy * 100).toFixed(2)}%</TableCell>
              </TableRow>
              <TableRow>
                <TableCell component="th" scope="row">Precision</TableCell>
                <TableCell align="right">{(train.precision * 100).toFixed(2)}%</TableCell>
                <TableCell align="right">{(test.precision * 100).toFixed(2)}%</TableCell>
              </TableRow>
              <TableRow>
                <TableCell component="th" scope="row">Recall</TableCell>
                <TableCell align="right">{(train.recall * 100).toFixed(2)}%</TableCell>
                <TableCell align="right">{(test.recall * 100).toFixed(2)}%</TableCell>
              </TableRow>
              <TableRow>
                <TableCell component="th" scope="row">F1 Score</TableCell>
                <TableCell align="right">{(train.f1 * 100).toFixed(2)}%</TableCell>
                <TableCell align="right">{(test.f1 * 100).toFixed(2)}%</TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
};

export default MetricsDashboard;
