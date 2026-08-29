/**
 * Reports watch progress at intervals and on unmount.
 *
 * Send completion ratio, not just "played". The backend converts it into the
 * implicit-feedback weight that trains collaborative filtering, so a 4-second
 * bounce and a full watch must not arrive looking identical.
 */
