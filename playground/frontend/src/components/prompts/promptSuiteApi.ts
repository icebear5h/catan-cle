export function errorMessage(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== 'object') {
    return fallback;
  }
  const record = payload as Record<string, unknown>;
  if (Array.isArray(record.errors) && record.errors.length > 0) {
    const first: unknown = record.errors[0];
    if (first && typeof first === 'object') {
      const error = first as Record<string, unknown>;
      const component = typeof error.component === 'string' ? error.component : 'suite';
      const message = typeof error.message === 'string' ? error.message : fallback;
      return `${component}: ${message}`;
    }
  }
  for (const key of ['details', 'message', 'error']) {
    if (typeof record[key] === 'string') {
      return record[key] as string;
    }
  }
  return fallback;
}

export async function readObject(response: Response): Promise<Record<string, unknown>> {
  const payload: unknown = await response.json();
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error(`Prompt Suite API returned invalid JSON (${response.status})`);
  }
  return payload as Record<string, unknown>;
}

export function caughtMessage(caught: unknown): string {
  return caught instanceof Error ? caught.message : String(caught);
}

/** Fetch a Prompt Suite endpoint and return its JSON object, throwing the API's own error message on failure. */
export async function fetchSuiteObject(
  url: string,
  init: RequestInit,
  failure: string,
): Promise<Record<string, unknown>> {
  const response = await fetch(url, init);
  const data = await readObject(response);
  if (!response.ok) {
    throw new Error(errorMessage(data, failure));
  }
  return data;
}

export interface RequestStatus {
  setBusy: (busy: boolean) => void;
  setError: (error: string | null) => void;
  setNotice: (notice: string | null) => void;
}

/**
 * Run one user-initiated Prompt Suite request: marks the Studio busy, clears
 * stale messages, reports the result's notice or the failure, and always
 * releases busy.
 */
export async function request(
  status: RequestStatus,
  url: string,
  init: RequestInit,
  failure: string,
  onSuccess: (data: Record<string, unknown>) => string | null,
): Promise<void> {
  try {
    status.setBusy(true);
    status.setError(null);
    status.setNotice(null);
    const data = await fetchSuiteObject(url, init, failure);
    const notice = onSuccess(data);
    if (notice !== null) status.setNotice(notice);
  } catch (caught) {
    status.setError(caughtMessage(caught));
  } finally {
    status.setBusy(false);
  }
}
