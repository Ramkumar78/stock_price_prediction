import { render, screen } from '@testing-library/react';
import MetricsDashboard from '../MetricsDashboard';
import { describe, it, expect } from 'vitest';

describe('MetricsDashboard', () => {
  const mockMetrics = {
    metrics: {
      train: {
        roc_auc: 0.9,
        accuracy: 0.85,
        precision: 0.8,
        recall: 0.75
      },
      test: {
        roc_auc: 0.85,
        accuracy: 0.8,
        precision: 0.75,
        recall: 0.7
      }
    }
  };

  it('renders dashboard title and metrics', () => {
    render(<MetricsDashboard metrics={mockMetrics} modelName="xgboost" />);
    expect(screen.getByText(/Performance Analysis/i)).toBeInTheDocument();
    expect(screen.getByText(/XGBOOST/i)).toBeInTheDocument();

    // Check for values (formatted)
    // Accuracy 0.8 -> 80.0%
    expect(screen.getByText(/80.0%/i)).toBeInTheDocument();
    // ROC AUC 0.85 -> 0.850
    expect(screen.getByText(/0.850/i)).toBeInTheDocument();
  });

  it('renders empty state when no metrics', () => {
    render(<MetricsDashboard metrics={null} modelName="xgboost" />);
    expect(screen.getByText(/No metrics found for xgboost/i)).toBeInTheDocument();
  });
});
