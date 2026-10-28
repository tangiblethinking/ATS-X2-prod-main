import { createRouter } from "@tanstack/react-router";
import { AppErrorComponent } from "@/lib/error-component";
import { routeTree } from "./routeTree.gen";

export function getRouter() {
  return createRouter({
    routeTree,
    defaultErrorComponent: AppErrorComponent,
    // Base path matches the /jobs prefix used by the domain router.
    basepath: "/jobs",
    trailingSlash: "never",
  });
}
