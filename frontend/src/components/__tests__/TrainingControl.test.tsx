import { render, screen, fireEvent } from '@testing-library/react';
import TrainingControl from '../TrainingControl';
import { describe, it, expect, vi } from 'vitest';

describe('TrainingControl', () => {
  const defaultProps = {
    selectedModel: 'xgboost',
    onModelChange: vi.fn(),
    onDownloadData: vi.fn(),
    onGenerateFeatures: vi.fn(),
    onTrainModel: vi.fn(),
    isDownloading: false,
    isGenerating: false,
    isTraining: false,
  };

  it('renders train button with correct model name', () => {
    render(<TrainingControl {...defaultProps} />);
    // The button text is "Train xgboost"
    expect(screen.getByRole('button', { name: /Train xgboost/i })).toBeInTheDocument();
  });

  it('calls onTrainModel on click', () => {
    render(<TrainingControl {...defaultProps} />);

    const button = screen.getByRole('button', { name: /Train xgboost/i });
    fireEvent.click(button);

    expect(defaultProps.onTrainModel).toHaveBeenCalled();
  });

  it('disables buttons when processing', () => {
    render(<TrainingControl {...defaultProps} isTraining={true} />);
    // Button text changes to "Training Model..."
    expect(screen.getByRole('button', { name: /Training Model.../i })).toBeDisabled();
  });
});
