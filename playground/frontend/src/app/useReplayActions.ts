import type { NativeReasoningEffort, ReplayLLMResponse } from '../types';
import {
  LIVE_MODEL_STORAGE_KEY,
  NATIVE_REASONING_STORAGE_KEY,
  REPLAY_MODEL_STORAGE_KEY,
  SERVER_URL,
  nativeReasoningRequest,
} from './constants';
import { getApiError, readApiObject } from './api';
import type { ReplayStepResult } from './sessionTypes';
import type { AppState } from './useAppState';
import type { SnapshotSync } from './useSnapshotSync';

// Colonist replay playback, model preferences, and replay LLM generation.
export function useReplayActions(state: AppState, sync: SnapshotSync) {
  const {
    replayModel, nativeReasoningEffort, historyRef, traceBrowseRequestRef,
    replayModelRef, replayCursorRef,
    setViewingHistory, setTraceBrowseBusy, setBrowsedTraceStep, setTraceBrowseError,
    setSelectedSavedGameId, setReplayLlmResponse, setReplayLlmError, setReplayPlaybackError,
    setIsReplayPlaybackProcessing, setHasActiveGame, setLastReplayStep, setLiveError,
    setLiveModel, setReplayModel, setNativeReasoningEffort, setIsReplayLlmProcessing,
  } = state;
  const { applyLiveStepError } = sync;

  const loadReplay = async (gameId: string) => {
    historyRef.current = false;
    setViewingHistory(false);
    traceBrowseRequestRef.current += 1;
    setTraceBrowseBusy(false);
    setBrowsedTraceStep(null);
    setTraceBrowseError(null);
    setSelectedSavedGameId('');
    setReplayLlmResponse(null);
    setReplayLlmError(null);
    setReplayPlaybackError(null);
    applyLiveStepError(null);
    setIsReplayPlaybackProcessing(true);
    try {
      console.log('[FRONTEND] Loading replay:', gameId);
      const response = await fetch(`${SERVER_URL}/api/load-replay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_id: gameId }),
      });
      const data = await readApiObject(response);
      if (!response.ok) {
        throw new Error(getApiError(data, 'Failed to load replay'));
      }
      setHasActiveGame(true);
      setLastReplayStep(null);
      console.log('Replay loaded:', data);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setReplayPlaybackError(message);
      setLiveError(message);
      console.error('Error loading replay:', error);
    } finally {
      setIsReplayPlaybackProcessing(false);
    }
  };

  const runReplayMutation = async (
    endpoint: string,
    init: RequestInit,
    fallbackError: string,
  ): Promise<Record<string, unknown> | null> => {
    setIsReplayPlaybackProcessing(true);
    setReplayPlaybackError(null);
    try {
      const response = await fetch(`${SERVER_URL}${endpoint}`, init);
      const data = await readApiObject(response);
      if (!response.ok) {
        throw new Error(getApiError(data, fallbackError));
      }
      return data;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setReplayPlaybackError(message);
      console.error(fallbackError, error);
      return null;
    } finally {
      setIsReplayPlaybackProcessing(false);
    }
  };

  const replayStep = async () => {
    const data = await runReplayMutation(
      '/api/replay-step',
      { method: 'POST' },
      'Failed to step replay',
    );
    if (data) {
      setLastReplayStep(data as unknown as ReplayStepResult);
      console.log('Replay step:', data);
    }
  };

  const replayUndo = async () => {
    const data = await runReplayMutation(
      '/api/replay-undo',
      { method: 'POST' },
      'Failed to move to the previous replay step',
    );
    if (data) {
      setLastReplayStep(null);
      console.log('Replay previous:', data);
    }
  };

  const setReplayStep = async (step: number) => {
    const data = await runReplayMutation(
      '/api/replay-goto-sequential',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ step }),
      },
      `Failed to go to replay step ${step}`,
    );
    if (data) {
      setLastReplayStep(null);
      console.log('Replay goto:', data);
    }
  };

  const runUntilDrift = async () => {
    const data = await runReplayMutation(
      '/api/replay-goto-divergence',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_steps: 500 }),
      },
      'Failed to run replay until drift',
    );
    if (data) {
      setLastReplayStep(null);
      console.log('Run until drift:', data);
    }
  };

  const updateLiveModel = (model: string) => {
    setLiveModel(model);
    window.localStorage.setItem(LIVE_MODEL_STORAGE_KEY, model);
  };

  const updateReplayModel = (model: string) => {
    replayModelRef.current = model;
    setReplayModel(model);
    window.localStorage.setItem(REPLAY_MODEL_STORAGE_KEY, model);
    setReplayLlmResponse(null);
    setReplayLlmError(null);
  };

  const updateNativeReasoningEffort = (effort: NativeReasoningEffort) => {
    setNativeReasoningEffort(effort);
    window.localStorage.setItem(NATIVE_REASONING_STORAGE_KEY, effort);
    setReplayLlmResponse(null);
    setReplayLlmError(null);
  };

  const generateReplayResponse = async () => {
    const requestCursor = replayCursorRef.current;
    const requestModel = replayModel.trim();
    if (!requestCursor) {
      setReplayLlmError('Navigate to a loaded replay position first.');
      return;
    }

    try {
      setIsReplayLlmProcessing(true);
      setReplayLlmError(null);

      const response = await fetch(`${SERVER_URL}/api/replay-llm-response`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: requestModel,
          reasoning: nativeReasoningRequest(nativeReasoningEffort),
        }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(getApiError(payload, 'Failed to generate replay response'));
      }

      const generated = payload as ReplayLLMResponse;
      const currentCursor = replayCursorRef.current;
      const clientDetectedStale = (
        currentCursor === null
        || currentCursor.gameId !== requestCursor.gameId
        || currentCursor.eventIndex !== requestCursor.eventIndex
        || replayModelRef.current.trim() !== requestModel
      );
      const result = {
        ...generated,
        stale: generated.stale || clientDetectedStale,
      };

      setReplayLlmResponse(result);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error('Error generating replay response:', error);
      setReplayLlmError(message);
    } finally {
      setIsReplayLlmProcessing(false);
    }
  };

  return {
    loadReplay, replayStep, replayUndo, setReplayStep, runUntilDrift,
    updateLiveModel, updateReplayModel, updateNativeReasoningEffort, generateReplayResponse,
  };
}

export type ReplayActions = ReturnType<typeof useReplayActions>;
