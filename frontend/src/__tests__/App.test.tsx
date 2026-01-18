import { render, screen, waitFor } from '@testing-library/react';
import App from '../App';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios from 'axios';

// Mock axios
vi.mock('axios');

describe('App', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('renders the main dashboard and fetches data', async () => {
    // Mock successful responses
    (axios.get as any).mockImplementation((url: string) => {
      if (url.includes('/metrics')) {
        return Promise.resolve({
          data: {
            metrics: {
              train: { roc_auc: 0.9, accuracy: 0.9 },
              test: { roc_auc: 0.8, accuracy: 0.8 }
            }
          }
        });
      }
      if (url.includes('/predict')) {
        return Promise.resolve({
            data: { prediction: 'UP', probability: 0.7 }
        });
      }
      return Promise.reject(new Error('not found'));
    });

    render(<App />);

    expect(screen.getByText(/SPY Price Prediction AI/i)).toBeInTheDocument();

    // Check if controls are rendered
    expect(screen.getByText(/Control Panel/i)).toBeInTheDocument();

    // Wait for async fetch
    await waitFor(() => {
       expect(axios.get).toHaveBeenCalledTimes(2); // Metrics and Prediction
    });
  });
});
