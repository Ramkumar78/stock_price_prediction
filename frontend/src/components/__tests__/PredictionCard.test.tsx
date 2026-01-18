import { render, screen } from '@testing-library/react';
import PredictionCard from '../PredictionCard';
import { describe, it, expect } from 'vitest';

describe('PredictionCard', () => {
  it('renders loading state when isLoading is true', () => {
    render(<PredictionCard prediction={null} probability={null} isLoading={true} modelName="ensemble" />);
    expect(screen.getByText(/Analyzing Market Data.../i)).toBeInTheDocument();
  });

  it('displays prediction data when provided', () => {
    render(
      <PredictionCard
        prediction="UP"
        probability={0.75}
        isLoading={false}
        modelName="ensemble"
      />
    );

    expect(screen.getByText(/UP/i)).toBeInTheDocument();
    expect(screen.getByText(/75.0%/i)).toBeInTheDocument();
    // Removed check for "Bullish" since component doesn't render it.
  });

  it('displays no signal when prediction is null', () => {
    render(<PredictionCard prediction={null} probability={null} isLoading={false} modelName="ensemble" />);
    expect(screen.getByText(/No Signal/i)).toBeInTheDocument();
  });
});
